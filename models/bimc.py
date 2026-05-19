import torch
import torch.nn as nn
import torch.nn.functional as F
import models.clip.clip as clip
import json
from utils.phase1_fusion import compute_query_reliability_beta, fuse_prototypes
from utils.phase2_prototypes import refine_description_prototypes, refine_visual_prototypes
from utils.phase3_scores import (
    apply_temperature,
    compute_dynamic_alpha,
    compute_hubness_bias,
    select_hubness_reference,
)
from utils.phase4_covariance import combine_novel_auxiliary, prepare_covariance
from utils.phase5_space import common_direction, remove_common_direction, transform_covariance
from utils.phase6_alignment import align_text_prototypes, apply_label_prior
from utils.phase7_separation import separate_prototypes

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

        phase2_opts = self.cfg.TRAINER.BiMC.PHASE2
        if phase2_opts.ENABLED:
            description_proto = refine_description_prototypes(
                description_features=description_features,
                description_targets=description_targets,
                description_proto=description_proto,
                text_proto=text_features,
                class_ids=class_index,
                mode=phase2_opts.TEXT_PROTO_MODE,
                reweight_tau=phase2_opts.TEXT_REWEIGHT_TAU,
                combine_weight=phase2_opts.TEXT_COMBINE_WEIGHT,
            )
        
        images_features, images_targets, images_proto = \
                                    self.inference_all_img_feature(loader, cls_begin_index)

        if cls_begin_index != 0:
            if phase2_opts.ENABLED and phase2_opts.VISUAL_PROTO_MODE != "original":
                print(f'phase2 visual proto refinement on class [{class_index}]')
                images_proto = refine_visual_prototypes(
                    visual_proto=images_proto,
                    base_visual_proto=self.base_vision_prototype,
                    support_features=images_features,
                    support_labels=images_targets,
                    class_ids=class_index,
                    mode=phase2_opts.VISUAL_PROTO_MODE,
                    shrinkage=phase2_opts.VISUAL_SHRINKAGE,
                    dynamic_lambda=phase2_opts.DYNAMIC_LAMBDA_I,
                    tau=self.cfg.TRAINER.BiMC.TAU,
                )
            elif calibrate_novel_vision_proto:
                print(f'calibrate vision proto on class [{class_index}]')
                images_proto = self.soft_calibration(self.base_vision_prototype, images_proto)
        else:
            self.base_vision_prototype = images_proto


        cov_images = torch.cov(images_features.T)

        if cls_begin_index == 0:
            cov_images = shrink_cov(cov_images, alpha1=self.cfg.TRAINER.BiMC.GAMMA_BASE) 
        else:
            cov_images = shrink_cov(cov_images, alpha1=self.cfg.TRAINER.BiMC.GAMMA_INC)

        phase4_opts = self.cfg.TRAINER.BiMC.PHASE4
        if phase4_opts.ENABLED:
            cov_images = prepare_covariance(
                cov_images,
                mode=phase4_opts.COV_MODE,
                hybrid_diag_weight=phase4_opts.HYBRID_DIAG_WEIGHT,
                eps=phase4_opts.EPS,
            )

        
        print('finish loading covariance')

        return {
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

   

    def forward_ours(self, images, num_cls, num_base_cls,
                           image_proto, cov_image,
                           description_proto,
                           description_features, description_targets,
                           text_features,
                           beta,
                           support_features=None,
                           support_labels=None,
                           return_beta_info=False):

        def knn_similarity_scores(queries, support_features, support_labels):
            """
            Compute the similarity between each query sample and all support samples,
            and retrieve the maximum score for each class per query.
            """
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


        img_feat = self.extract_img_feature(images)
        img_feat = F.normalize(img_feat, dim=-1)

        text_proto = self.calibrated_text_proto(text_features, description_proto)
        phase6_opts = self.cfg.TRAINER.BiMC.PHASE6
        phase7_opts = self.cfg.TRAINER.BiMC.PHASE7
        phase5_opts = self.cfg.TRAINER.BiMC.PHASE5
        prototype_phase_enabled = phase6_opts.ENABLED or phase7_opts.ENABLED or phase5_opts.ENABLED
        if prototype_phase_enabled:
            image_proto = F.normalize(image_proto, dim=-1)
            text_proto = F.normalize(text_proto, dim=-1)
            description_proto = F.normalize(description_proto, dim=-1)
            description_features = F.normalize(description_features, dim=-1)

        if phase6_opts.ENABLED:
            text_proto = align_text_prototypes(
                text_proto,
                image_proto,
                mode=phase6_opts.ALIGNMENT_MODE,
                strength=phase6_opts.ALIGNMENT_STRENGTH,
                ot_tau=phase6_opts.OT_TAU,
            )

        if phase7_opts.ENABLED:
            image_proto = separate_prototypes(
                image_proto,
                mode=phase7_opts.SEPARATION_MODE,
                strength=phase7_opts.SEPARATION_STRENGTH,
                graph_k=phase7_opts.GRAPH_K,
            )
            text_proto = separate_prototypes(
                text_proto,
                mode=phase7_opts.SEPARATION_MODE,
                strength=phase7_opts.SEPARATION_STRENGTH,
                graph_k=phase7_opts.GRAPH_K,
            )

        if phase5_opts.ENABLED and phase5_opts.SPACE_TRANSFORM == "common_direction_removal":
            direction = common_direction(
                [image_proto, text_proto, description_proto, description_features],
                eps=phase5_opts.EPS,
            )
            if phase5_opts.APPLY_TO in ("all", "queries"):
                img_feat = remove_common_direction(img_feat, direction, strength=phase5_opts.STRENGTH)
            if phase5_opts.APPLY_TO in ("all", "prototypes"):
                image_proto = remove_common_direction(image_proto, direction, strength=phase5_opts.STRENGTH)
                text_proto = remove_common_direction(text_proto, direction, strength=phase5_opts.STRENGTH)
                description_proto = remove_common_direction(description_proto, direction, strength=phase5_opts.STRENGTH)
                description_features = remove_common_direction(description_features, direction, strength=phase5_opts.STRENGTH)
                cov_image = transform_covariance(cov_image, direction, strength=phase5_opts.STRENGTH, eps=phase5_opts.EPS)

        phase1_opts = self.cfg.TRAINER.BiMC
        beta_info = {
            "mode": phase1_opts.FUSION_BETA_MODE,
            "geometry": phase1_opts.FUSION_GEOMETRY,
            "beta": None,
        }

        if phase1_opts.FUSION_BETA_MODE == "fixed" and phase1_opts.FUSION_GEOMETRY == "linear":
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

        phase3_opts = self.cfg.TRAINER.BiMC.PHASE3
        if phase3_opts.ENABLED and phase3_opts.HUBNESS_ENABLED:
            hub_reference = select_hubness_reference(
                phase3_opts.HUBNESS_SOURCE,
                support_features=support_features,
                description_features=description_features,
            )
            hub_proto = fused_proto if fused_proto.ndim == 2 else fused_proto.mean(dim=0)
            hub_bias = compute_hubness_bias(
                hub_reference,
                hub_proto,
                strength=phase3_opts.HUBNESS_STRENGTH,
            )
            logits_proto_fused = logits_proto_fused - hub_bias.view(1, -1)

        logits_proto_scaled = apply_temperature(
            logits_proto_fused,
            enabled=phase3_opts.ENABLED and phase3_opts.TEMP_SCALING_ENABLED,
            temperature=phase3_opts.TEMP_VALUE,
        )
        prob_fused_proto = F.softmax(logits_proto_scaled, dim=-1)

        logits_cov = _cov_forward(img_feat, image_proto, cov_image)
        logits_knn = knn_similarity_scores(img_feat, description_features, description_targets)
        logits_cov_scaled = apply_temperature(
            logits_cov / 512,
            enabled=phase3_opts.ENABLED and phase3_opts.TEMP_SCALING_ENABLED,
            temperature=phase3_opts.TEMP_VALUE,
        )
        logits_knn_scaled = apply_temperature(
            logits_knn,
            enabled=phase3_opts.ENABLED and phase3_opts.TEMP_SCALING_ENABLED,
            temperature=phase3_opts.TEMP_VALUE,
        )
        prob_cov = F.softmax(logits_cov_scaled, dim=-1)
        prob_knn = F.softmax(logits_knn_scaled, dim=-1)

        NUM_BASE_CLS = num_base_cls
        use_diversity = self.cfg.TRAINER.BiMC.USING_ENSEMBLE
        if use_diversity:
            ensemble_alpha = self.cfg.DATASET.ENSEMBLE_ALPHA
        else:
            ensemble_alpha = 1.0

        phase4_opts = self.cfg.TRAINER.BiMC.PHASE4
        novel_aux_probs = prob_knn[:, NUM_BASE_CLS:]
        novel_aux_logits = logits_knn_scaled[:, NUM_BASE_CLS:]
        if phase4_opts.ENABLED and phase4_opts.APPLY_TO_NOVEL and prob_knn.shape[1] > NUM_BASE_CLS:
            novel_aux_probs = combine_novel_auxiliary(
                prob_knn[:, NUM_BASE_CLS:],
                prob_cov[:, NUM_BASE_CLS:],
                replace_novel_nn=phase4_opts.REPLACE_NOVEL_NN,
                combine_mode=phase4_opts.NOVEL_AUX_COMBINE_MODE,
            )
            if phase4_opts.REPLACE_NOVEL_NN:
                novel_aux_logits = logits_cov_scaled[:, NUM_BASE_CLS:]
            elif phase4_opts.NOVEL_AUX_COMBINE_MODE == "average":
                novel_aux_logits = 0.5 * (logits_knn_scaled[:, NUM_BASE_CLS:] + logits_cov_scaled[:, NUM_BASE_CLS:])

        aux_probs = torch.cat([prob_cov[:, :NUM_BASE_CLS], novel_aux_probs], dim=1)
        aux_logits = torch.cat([logits_cov_scaled[:, :NUM_BASE_CLS], novel_aux_logits], dim=1)

        if phase3_opts.ENABLED and phase3_opts.DYNAMIC_ALPHA_ENABLED:
            alpha = compute_dynamic_alpha(
                logits_proto_scaled,
                aux_logits,
                mode=phase3_opts.DYNAMIC_ALPHA_MODE,
                alpha_clip_min=phase3_opts.ALPHA_CLIP_MIN,
                alpha_clip_max=phase3_opts.ALPHA_CLIP_MAX,
                energy_enabled=phase3_opts.ENERGY_ENABLED,
            ).view(-1, 1)
        else:
            alpha = ensemble_alpha

        base_probs = alpha * prob_fused_proto[:, :NUM_BASE_CLS] + (1 - alpha) * aux_probs[:, :NUM_BASE_CLS]
        inc_probs = alpha * prob_fused_proto[:, NUM_BASE_CLS:] + (1 - alpha) * aux_probs[:, NUM_BASE_CLS:]

        prob_fused = torch.cat([base_probs, inc_probs], dim=1)
        if phase6_opts.ENABLED and phase6_opts.LABEL_PRIOR_ENABLED:
            prob_fused = apply_label_prior(
                prob_fused,
                enabled=phase6_opts.LABEL_PRIOR_ENABLED,
                transductive=phase6_opts.LABEL_PRIOR_TRANSDUCTIVE,
                strength=phase6_opts.LABEL_PRIOR_STRENGTH,
            )
        logits = prob_fused
        if return_beta_info:
            return logits, beta_info
        return logits



    @torch.no_grad()
    def extract_img_feature(self, images):
        images = images.to(self.device)
        image_features = self.clip_model.encode_image(images)
        return image_features


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
