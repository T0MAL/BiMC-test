import torch
import torch.nn as nn
import torch.nn.functional as F
import models.clip.clip as clip
import json
from utils.phase1_fusion import (
    cnn_uses_proto_adjust,
    cnn_uses_query_beta,
    compute_cnn_query_reliability_beta,
    compute_query_reliability_beta,
    fuse_prototypes,
    get_active_cnn_mode,
    project_cnn_features_to_clip,
)

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGENET_IMAGE_STD = (0.229, 0.224, 0.225)


class BiMC(nn.Module):

    def __init__(self, cfg, template, device):
        super(BiMC, self).__init__()
        self.cfg = cfg
        self.device = device
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        print(f"Prompt template:{template}")
        self.template = template
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.BiMC.PREC == "fp32" or cfg.TRAINER.BiMC.PREC == "amp":
        # CLIP's default precision is fp16
            clip_model.float()

        clip_model.eval()
        self.clip_model = clip_model.to(self.device)
        self.text_proto = None
        self.description_proto = None
        self.vision_proto = None
        self.cnn_feature_extractor = None
        self.cnn_feature_dim = None
        self._cnn_projection_matrices = {}
        self._cnn_support_cache = {}
        self._init_cnn_branch()


    def _cnn_opts(self):
        return self.cfg.TRAINER.BiMC


    def _active_cnn_mode(self):
        return get_active_cnn_mode(self._cnn_opts())


    def _cnn_enabled(self):
        return self._active_cnn_mode() != "none"


    def _init_cnn_branch(self):
        opts = self._cnn_opts()
        mode = self._active_cnn_mode()
        print(
            "CNN branch: "
            f"mode={mode}, backbone={opts.CNN_BACKBONE}, lambda={opts.CNN_LAMBDA}, "
            f"topk={opts.CNN_TOPK}, projection={opts.CNN_PROJECTION}, "
            f"cache_features={opts.CNN_CACHE_FEATURES}"
        )
        if mode == "none":
            return

        self.cnn_feature_extractor, self.cnn_feature_dim = self._build_cnn_backbone(opts.CNN_BACKBONE)
        self.cnn_feature_extractor = self.cnn_feature_extractor.to(self.device)
        self.cnn_feature_extractor.eval()
        for param in self.cnn_feature_extractor.parameters():
            param.requires_grad_(False)
        print(f"Frozen CNN backbone ready: {opts.CNN_BACKBONE}, feature_dim={self.cnn_feature_dim}")


    def _build_cnn_backbone(self, backbone_name):
        try:
            import torchvision.models as tv_models
        except ImportError as exc:
            raise ImportError(
                "CNN BiMC modes require torchvision. Install torchvision or set "
                "CNN_EXPERIMENT_MODE=none."
            ) from exc

        constructor = getattr(tv_models, backbone_name, None)
        if constructor is None:
            raise ValueError(f"Unsupported torchvision CNN backbone: {backbone_name}")

        weights_cls_name = self._torchvision_weights_class_name(backbone_name)
        weights_cls = getattr(tv_models, weights_cls_name, None)
        attempts = []
        if weights_cls is not None:
            weights = getattr(weights_cls, "DEFAULT", None)
            if weights is not None:
                attempts.append(("pretrained weights", {"weights": weights}))
        attempts.append(("legacy pretrained=True", {"pretrained": True}))
        attempts.append(("random initialization fallback", {}))

        last_error = None
        model = None
        for label, kwargs in attempts:
            try:
                model = constructor(**kwargs)
                print(f"Loaded {backbone_name} with {label}.")
                break
            except TypeError as exc:
                last_error = exc
            except Exception as exc:
                last_error = exc
                print(f"Could not load {backbone_name} with {label}: {exc}")

        if model is None:
            raise RuntimeError(f"Failed to construct CNN backbone {backbone_name}: {last_error}")

        if hasattr(model, "fc") and hasattr(model.fc, "in_features"):
            feature_dim = model.fc.in_features
            model.fc = nn.Identity()
            return model, feature_dim

        raise ValueError(
            f"Backbone {backbone_name} is not supported as a feature extractor. "
            "Use a torchvision ResNet-style model with an fc layer."
        )


    def _torchvision_weights_class_name(self, backbone_name):
        if backbone_name.startswith("resnet"):
            suffix = backbone_name.replace("resnet", "")
            return f"ResNet{suffix}_Weights"
        parts = backbone_name.split("_")
        return "".join(part[:1].upper() + part[1:] for part in parts) + "_Weights"


    @torch.no_grad()
    def inference_text_feature(self, class_names, template, cls_begin_index):
        print(f'class names: {class_names}')
        clip_weights = []
        all_targets = []
        k = cls_begin_index
        for classname in class_names:
            targets = torch.full((len(template),), k)
            all_targets.append(targets)
            k += 1
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            classname = classname.replace('-', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = self.clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)
        clip_weights = torch.stack(clip_weights, dim=0)
        clip_weights = F.normalize(clip_weights, dim=-1)
        all_targets = torch.cat(all_targets, dim=0)
        return clip_weights, all_targets


    @torch.no_grad()
    def inference_all_img_feature(self, loader, cls_begin_index):
        all_features = []
        all_labels = []
        for batch in loader:
            images, labels = self.parse_batch(batch)
            features = self.clip_model.encode_image(images)
            features = F.normalize(features, dim=-1)
            all_features.append(features)
            all_labels.append(labels)
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        unique_labels = torch.unique(all_labels)
        print(f'all targets:{unique_labels}')
        prototypes = []
        for c in unique_labels:
            idx = torch.where(c == all_labels)[0]
            class_features = all_features[idx]
            class_prototype = class_features.mean(dim=0)
            prototypes.append(class_prototype)
        prototypes = torch.stack(prototypes, dim=0)
        prototypes = F.normalize(prototypes, dim=-1)
        return all_features, all_labels, prototypes


    @torch.no_grad()
    def inference_all_cnn_feature(self, loader, cls_begin_index):
        cache_key = None
        if self._cnn_opts().CNN_CACHE_FEATURES:
            cache_key = (id(loader.dataset), int(cls_begin_index), self._cnn_opts().CNN_BACKBONE)
            if cache_key in self._cnn_support_cache:
                return self._cnn_support_cache[cache_key]

        all_features = []
        all_labels = []
        for batch in loader:
            images, labels = self.parse_batch(batch)
            features = self.extract_cnn_feature(images)
            all_features.append(features)
            all_labels.append(labels)
        all_features = torch.cat(all_features, dim=0)
        all_features = F.normalize(all_features, dim=-1)
        all_labels = torch.cat(all_labels, dim=0)

        unique_labels = torch.unique(all_labels)
        prototypes = []
        for c in unique_labels:
            idx = torch.where(c == all_labels)[0]
            class_features = all_features[idx]
            class_prototype = class_features.mean(dim=0)
            prototypes.append(class_prototype)
        prototypes = torch.stack(prototypes, dim=0)
        prototypes = F.normalize(prototypes, dim=-1)

        result = (all_features, all_labels, prototypes)
        if cache_key is not None:
            self._cnn_support_cache[cache_key] = result
        return result


    def _project_cnn_prototypes(self, cnn_proto, clip_dim):
        opts = self._cnn_opts()
        key = (cnn_proto.shape[-1], clip_dim, opts.CNN_PROJECTION)
        projection_matrix = self._cnn_projection_matrices.get(key)
        projected, projection_matrix = project_cnn_features_to_clip(
            cnn_proto,
            clip_dim=clip_dim,
            projection=opts.CNN_PROJECTION,
            seed=0,
            projection_matrix=projection_matrix,
            return_matrix=True,
        )
        self._cnn_projection_matrices[key] = projection_matrix.detach()
        return projected


    def _apply_cnn_proto_adjustment(self, image_proto, cnn_proto):
        lambda_cnn = self._cnn_opts().CNN_LAMBDA
        cnn_projected = self._project_cnn_prototypes(cnn_proto, image_proto.shape[-1])
        adjusted_proto = F.normalize(
            (1 - lambda_cnn) * F.normalize(image_proto, dim=-1)
            + lambda_cnn * cnn_projected,
            dim=-1,
        )

        cosine_projected = (F.normalize(image_proto, dim=-1) * cnn_projected).sum(dim=-1)
        cosine_adjusted = (F.normalize(image_proto, dim=-1) * adjusted_proto).sum(dim=-1)
        delta_norm = (adjusted_proto - F.normalize(image_proto, dim=-1)).norm(dim=-1)
        stats = {
            "cnn_projected_cos_mean": float(cosine_projected.mean().item()),
            "cnn_projected_cos_min": float(cosine_projected.min().item()),
            "cnn_adjusted_cos_mean": float(cosine_adjusted.mean().item()),
            "cnn_adjusted_delta_mean": float(delta_norm.mean().item()),
            "cnn_adjusted_delta_max": float(delta_norm.max().item()),
        }
        print(
            "CNN prototype adjustment stats: "
            f"lambda={lambda_cnn:.4f}, "
            f"projected_cos_mean={stats['cnn_projected_cos_mean']:.4f}, "
            f"projected_cos_min={stats['cnn_projected_cos_min']:.4f}, "
            f"adjusted_cos_mean={stats['cnn_adjusted_cos_mean']:.4f}, "
            f"delta_mean={stats['cnn_adjusted_delta_mean']:.4f}, "
            f"delta_max={stats['cnn_adjusted_delta_max']:.4f}"
        )
        return adjusted_proto, cnn_projected, stats


    @torch.no_grad()
    def inference_all_description_feature(self, class_names, gpt_path, cls_begin_index):
        description_embeddings = []
        mean_embeddings = []
        all_targets = []
        file = open(gpt_path, "r")
        GPT_prompt_dict = json.load(file)
        # The order of embeddings should follow strictly order of classname variable
        # Keys name should match classnames so that we could do fetching from the dict.
        # Convert the dict to lower case
        GPT_prompt_dict = {k.lower().replace("_", " "): v for k, v in GPT_prompt_dict.items()}
        k = cls_begin_index
        for single_key in class_names:
            single_class_prompts = GPT_prompt_dict[single_key.lower().replace("_", " ")]
            targets = torch.full((len(single_class_prompts),), k)

            k += 1
            x_tokenized = torch.cat([clip.tokenize(p) for p in single_class_prompts])
            with torch.no_grad():
                text_features = self.clip_model.encode_text(x_tokenized.cuda())
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_embeddings.append(text_features.mean(0).unsqueeze(0))
            description_embeddings.append(text_features)
            all_targets.append(targets)
        description_embeddings = torch.cat(description_embeddings, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        mean_embeddings = torch.cat(mean_embeddings, dim=0)
        mean_embeddings = F.normalize(mean_embeddings, dim=-1)
        return description_embeddings, all_targets, mean_embeddings


    def soft_calibration(self, base_protos, cur_protos):
        shift_weight = self.cfg.TRAINER.BiMC.LAMBDA_I
        tau = self.cfg.TRAINER.BiMC.TAU
        base_protos = F.normalize(base_protos, p=2, dim=-1)
        cur_protos = F.normalize(cur_protos, p=2, dim=-1)
        weights = torch.mm(cur_protos, base_protos.T) * tau
        norm_weights = torch.softmax(weights, dim=1)
        delta_protos = torch.matmul(norm_weights, base_protos)
        delta_protos = F.normalize(delta_protos, p=2, dim=-1)
        updated_protos = (1 - shift_weight) * cur_protos + shift_weight * delta_protos
        updated_protos = F.normalize(updated_protos, dim=-1)
        return updated_protos


    def calibrated_text_proto(self, text_features, description_proto):
        if self.cfg.TRAINER.BiMC.TEXT_CALIBRATION:
            lambda_t = self.cfg.TRAINER.BiMC.LAMBDA_T
        else:
            lambda_t = 0.0
        return (1 - lambda_t) * text_features + lambda_t * description_proto
    

    def build_task_statistics(self, class_names, loader,
                         class_index, calibrate_novel_vision_proto=False):
        
            
        def shrink_cov(cov, alpha1=1.0, alpha2=0.0):
            diag_mean = torch.mean(torch.diagonal(cov))
            off_diag = cov.clone()
            off_diag.fill_diagonal_(0.0)
            mask = off_diag != 0.0
            off_diag_mean = (off_diag*mask).sum() / mask.sum()
            iden = torch.eye(cov.shape[0]).to(cov.device)
            cov_ = cov + (alpha1*diag_mean*iden) + (alpha2*off_diag_mean*(1-iden))
            return cov_


        cls_begin_index = class_index[0]


        text_features, text_targets = self.inference_text_feature(class_names, self.template, cls_begin_index)

        description_features, description_targets, description_proto = \
                                  self.inference_all_description_feature(class_names=class_names, 
                                  gpt_path=self.cfg.DATASET.GPT_PATH,
                                  cls_begin_index=cls_begin_index)
        
        images_features, images_targets, images_proto = \
                                    self.inference_all_img_feature(loader, cls_begin_index)

        if cls_begin_index != 0:
            if calibrate_novel_vision_proto:
                print(f'calibrate vision proto on class [{class_index}]')
                images_proto = self.soft_calibration(self.base_vision_prototype, images_proto)
        else:
            self.base_vision_prototype = images_proto

        cnn_features = None
        cnn_targets = None
        cnn_proto = None
        cnn_projected_proto = None
        cnn_adjustment_stats = None
        cnn_mode = self._active_cnn_mode()
        if self._cnn_enabled():
            cnn_features, cnn_targets, cnn_proto = self.inference_all_cnn_feature(loader, cls_begin_index)
            print(
                "CNN support stats: "
                f"mode={cnn_mode}, samples={cnn_features.shape[0]}, "
                f"classes={cnn_proto.shape[0]}, feature_dim={cnn_features.shape[-1]}"
            )
            if cnn_uses_proto_adjust(cnn_mode):
                images_proto, cnn_projected_proto, cnn_adjustment_stats = \
                    self._apply_cnn_proto_adjustment(images_proto, cnn_proto)

        cov_images = torch.cov(images_features.T)

        if cls_begin_index == 0:
            cov_images = shrink_cov(cov_images, alpha1=self.cfg.TRAINER.BiMC.GAMMA_BASE) 
        else:
            cov_images = shrink_cov(cov_images, alpha1=self.cfg.TRAINER.BiMC.GAMMA_INC)

        
        print('finish loading covariance')

        task_statistics = {
            'description_proto': description_proto,
            'description_features': description_features,
            'description_targets': description_targets,

            'text_features': text_features,
            'text_targets': text_targets,           
  
            'image_proto': images_proto,
            'images_features': images_features,
            'images_targets': images_targets,
            'cov_image': cov_images,
            
            'class_index': class_index,
            'sample_cnt': len(images_features)
        }
        if self._cnn_enabled():
            task_statistics.update({
                'cnn_features': cnn_features,
                'cnn_targets': cnn_targets,
                'cnn_proto': cnn_proto,
            })
            if cnn_projected_proto is not None:
                task_statistics['cnn_projected_proto'] = cnn_projected_proto
            if cnn_adjustment_stats is not None:
                task_statistics['cnn_adjustment_stats'] = cnn_adjustment_stats
        return task_statistics

   

    def forward_ours(self, images, num_cls, num_base_cls,
                           image_proto, cov_image,
                           description_proto,
                           description_features, description_targets,
                           text_features,
                           beta,
                           cnn_proto=None,
                           return_beta_info=False):
    
        def knn_similarity_scores(queries, support_features, support_labels):
            """
            Compute the similarity between each query sample and all support samples,
            and retrieve the maximum score for each class per query.
            """
            # Ensure all inputs are on the same device
            device = queries.device
            support_features = support_features.to(device)
            support_labels = support_labels.to(device)
            similarity_scores = torch.matmul(queries, support_features.T)
            k = torch.max(support_labels) + 1
            max_scores = torch.full((queries.size(0), k), float('-inf'), device=device)
            expanded_labels = support_labels.unsqueeze(0).expand(queries.size(0), -1)
            for label in range(k):
                label_mask = (expanded_labels == label)
                masked_scores = similarity_scores.masked_fill(~label_mask, float('-inf'))
                max_scores[:, label] = torch.max(masked_scores, dim=1).values
            return max_scores


        def _mahalanobis(dist, cov_inv):
            """
            Compute the Mahalanobis distance between feature vectors and a class prototype.
            """
            left_term = torch.matmul(dist, cov_inv)
            mahal = torch.matmul(left_term, dist.T)
            return torch.diag(mahal)


        def _cov_forward(feat, proto, cov):
            """
            Perform a forward pass computing negative Mahalanobis distance between 
            features and each class prototype using a shared covariance matrix.
            """
            maha_dist = []
            inv_covmat = torch.pinverse(cov.to(dtype=torch.float32))
            inv_covmat = inv_covmat.to(dtype=proto.dtype)
            for cl in range(num_cls):
                distance = feat - proto[cl]
                dist = _mahalanobis(distance, inv_covmat)
                maha_dist.append(dist)
            maha_dist = torch.stack(maha_dist)
            logits = -maha_dist.T
            return logits
        

        # Normalize the image features
        img_feat = self.extract_img_feature(images)
        img_feat = F.normalize(img_feat, dim=-1)

        text_proto = self.calibrated_text_proto(text_features, description_proto)
        phase1_opts = self.cfg.TRAINER.BiMC
        cnn_mode = self._active_cnn_mode()
        beta_info = {
            "mode": phase1_opts.FUSION_BETA_MODE,
            "geometry": phase1_opts.FUSION_GEOMETRY,
            "cnn_mode": cnn_mode,
            "beta": None,
        }

        # Preserve the original BiMC formula exactly for the default baseline.
        if cnn_uses_query_beta(cnn_mode):
            if cnn_proto is None:
                raise ValueError(f"CNN prototypes are required for {cnn_mode}.")
            text_proto = F.normalize(text_proto, dim=-1)
            image_proto = F.normalize(image_proto, dim=-1)
            cnn_query_feat = self.extract_cnn_feature(images)
            beta_x, cnn_beta_info = compute_cnn_query_reliability_beta(
                query_clip_features=img_feat,
                text_proto=text_proto,
                query_cnn_features=cnn_query_feat,
                cnn_proto=cnn_proto,
                reliability_mode=phase1_opts.RELIABILITY_MODE,
                beta_clip_min=phase1_opts.BETA_CLIP_MIN,
                beta_clip_max=phase1_opts.BETA_CLIP_MAX,
                cnn_topk=phase1_opts.CNN_TOPK,
                return_info=True,
            )
            fused_proto = fuse_prototypes(
                image_proto.unsqueeze(0),
                text_proto.unsqueeze(0),
                beta_x.view(-1, 1, 1),
                geometry=phase1_opts.FUSION_GEOMETRY,
            )
            logits_proto_fused = torch.einsum("bd,bcd->bc", img_feat, fused_proto)
            beta_info["beta"] = beta_x.detach()
            beta_info["cnn_reliability_text"] = cnn_beta_info["reliability_text"]
            beta_info["cnn_reliability_visual"] = cnn_beta_info["reliability_visual"]
        elif phase1_opts.FUSION_BETA_MODE == "fixed" and phase1_opts.FUSION_GEOMETRY == "linear":
            fused_proto = beta * text_proto + (1 - beta) * image_proto
            fused_proto = F.normalize(fused_proto, dim=-1)
            logits_proto_fused = img_feat @ fused_proto.t()
        else:
            text_proto = F.normalize(text_proto, dim=-1)
            image_proto = F.normalize(image_proto, dim=-1)

            if phase1_opts.FUSION_BETA_MODE == "query_reliability":
                beta_x = compute_query_reliability_beta(
                    img_feat,
                    text_proto,
                    image_proto,
                    reliability_mode=phase1_opts.RELIABILITY_MODE,
                    beta_clip_min=phase1_opts.BETA_CLIP_MIN,
                    beta_clip_max=phase1_opts.BETA_CLIP_MAX,
                )
                fused_proto = fuse_prototypes(
                    image_proto.unsqueeze(0),
                    text_proto.unsqueeze(0),
                    beta_x.view(-1, 1, 1),
                    geometry=phase1_opts.FUSION_GEOMETRY,
                )
                logits_proto_fused = torch.einsum("bd,bcd->bc", img_feat, fused_proto)
                beta_info["beta"] = beta_x.detach()
            else:
                fused_proto = fuse_prototypes(
                    image_proto,
                    text_proto,
                    beta,
                    geometry=phase1_opts.FUSION_GEOMETRY,
                )
                logits_proto_fused = img_feat @ fused_proto.t()
                if isinstance(beta, torch.Tensor):
                    beta_info["beta"] = beta.detach()
        prob_fused_proto = F.softmax(logits_proto_fused, dim=-1)

        logits_cov = _cov_forward(img_feat, image_proto, cov_image)
        logits_knn = knn_similarity_scores(img_feat, description_features, description_targets)    
        prob_cov = F.softmax(logits_cov / 512, dim=-1)
        prob_knn = F.softmax(logits_knn, dim=-1)

        NUM_BASE_CLS = num_base_cls
        use_diversity = self.cfg.TRAINER.BiMC.USING_ENSEMBLE
        if use_diversity:
            ensemble_alpha = self.cfg.DATASET.ENSEMBLE_ALPHA
        else:
            ensemble_alpha = 1.0

        base_probs = ensemble_alpha * prob_fused_proto[:, :NUM_BASE_CLS] + (1 - ensemble_alpha) * prob_cov[:, :NUM_BASE_CLS]
        inc_probs = ensemble_alpha * prob_fused_proto[:, NUM_BASE_CLS:] + (1 - ensemble_alpha) * prob_knn[:, NUM_BASE_CLS:]

        prob_fused = torch.cat([base_probs, inc_probs], dim=1)
        logits = prob_fused
        if return_beta_info:
            return logits, beta_info
        return logits



    @torch.no_grad()
    def extract_img_feature(self, images):
        images = images.to(self.device)
        image_features = self.clip_model.encode_image(images)
        return image_features


    def _cnn_preprocess_images(self, images):
        clip_mean = torch.tensor(CLIP_IMAGE_MEAN, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        clip_std = torch.tensor(CLIP_IMAGE_STD, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        imagenet_mean = torch.tensor(IMAGENET_IMAGE_MEAN, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        imagenet_std = torch.tensor(IMAGENET_IMAGE_STD, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        images_01 = images * clip_std + clip_mean
        images_01 = images_01.clamp(0, 1)
        return (images_01 - imagenet_mean) / imagenet_std


    @torch.no_grad()
    def extract_cnn_feature(self, images):
        if self.cnn_feature_extractor is None:
            raise RuntimeError("CNN feature extractor is not initialized.")
        images = images.to(self.device)
        cnn_images = self._cnn_preprocess_images(images)
        features = self.cnn_feature_extractor(cnn_images)
        if isinstance(features, (tuple, list)):
            features = features[0]
        features = torch.flatten(features, start_dim=1)
        return F.normalize(features, dim=-1)


    @torch.no_grad()
    def forward(self, images):
        img_feat = self.extract_img_feature(images)
        img_feat = F.normalize(img_feat, dim=-1)
        classifier = F.normalize(self.classifier_weights, dim=-1)
        logits = 100. * img_feat @ classifier.t()
        return logits



    def parse_batch(self, batch):
        data = batch['image']
        targets = batch['label']
        data = data.to(self.device)
        targets = targets.to(self.device)
        return data, targets
