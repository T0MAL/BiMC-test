import torch
import torch.nn as nn
import torch.nn.functional as F
import models.clip.clip as clip
import json
from utils.phase1_fusion import compute_query_reliability_beta, fuse_prototypes
from utils.phase3_scores import (
    apply_hubness_correction,
    combine_with_dynamic_alpha,
    compute_prototype_hubness,
    dynamic_alpha_from_scores,
    energy_score,
    tensor_stats as phase3_tensor_stats,
)
from utils.phase4_covariance import mahalanobis_diag_score

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


def _clamped_weight(value):
    return min(1.0, max(0.0, float(value)))


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
        
        images_features, images_targets, images_proto = \
                                    self.inference_all_img_feature(loader, cls_begin_index)

        if cls_begin_index != 0:
            if calibrate_novel_vision_proto:
                print(f'calibrate vision proto on class [{class_index}]')
                images_proto = self.soft_calibration(self.base_vision_prototype, images_proto)
        else:
            self.base_vision_prototype = images_proto


        cov_images = torch.cov(images_features.T)

        if cls_begin_index == 0:
            cov_images = shrink_cov(cov_images, alpha1=self.cfg.TRAINER.BiMC.GAMMA_BASE) 
        else:
            cov_images = shrink_cov(cov_images, alpha1=self.cfg.TRAINER.BiMC.GAMMA_INC)

        
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
                           phase4_diag_vars=None,
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


        def _softmax_score(scores, temp):
            return F.softmax(scores / max(float(temp), 1e-8), dim=-1)


        def _phase3_temp(name):
            p3 = self.cfg.TRAINER.BiMC.PHASE3
            if name == "calib":
                return p3.TEMP_CALIB
            if name == "cov":
                return p3.TEMP_COV
            if name == "nn":
                return p3.TEMP_NN
            return 1.0


        def _score_probabilities(scores, name):
            p3 = self.cfg.TRAINER.BiMC.PHASE3
            if p3.ENABLED and p3.TEMP_SCALING_ENABLED:
                return _softmax_score(scores, _phase3_temp(name))
            return F.softmax(scores, dim=-1)


        def _record_energy_stats(info, prefix, scores):
            p3 = self.cfg.TRAINER.BiMC.PHASE3
            energy = energy_score(scores, temp=p3.ENERGY_TEMP)
            stats = phase3_tensor_stats(energy, prefix=f"energy_{prefix}")
            info["phase3"].update(stats)


        def _slice_alpha(
            p_calib,
            p_aux,
            s_calib,
            s_aux,
            split_name,
            temp_aux,
            info,
        ):
            p3 = self.cfg.TRAINER.BiMC.PHASE3
            if s_calib.shape[1] == 0:
                return p_calib
            alpha_x, alpha_stats = dynamic_alpha_from_scores(
                s_calib,
                s_aux,
                mode=p3.DYNAMIC_ALPHA_MODE,
                temp_calib=p3.TEMP_CALIB if p3.TEMP_SCALING_ENABLED else 1.0,
                temp_aux=temp_aux if p3.TEMP_SCALING_ENABLED else 1.0,
                min_alpha=p3.ALPHA_CLIP_MIN,
                max_alpha=p3.ALPHA_CLIP_MAX,
                eps=p3.ALPHA_EPS,
            )
            alpha_stats["split"] = split_name
            info["dynamic_alpha"].append(alpha_stats)
            return combine_with_dynamic_alpha(p_calib, p_aux, alpha_x)
        

        # Normalize the image features
        img_feat = self.extract_img_feature(images)
        img_feat = F.normalize(img_feat, dim=-1)

        text_proto = self.calibrated_text_proto(text_features, description_proto)
        phase1_opts = self.cfg.TRAINER.BiMC
        beta_info = {
            "mode": phase1_opts.FUSION_BETA_MODE,
            "geometry": phase1_opts.FUSION_GEOMETRY,
            "beta": None,
            "phase3": {},
            "phase4": {},
            "hubness_by_class": [],
            "dynamic_alpha": [],
        }

        # Preserve the original BiMC formula exactly for the default baseline.
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

        p3 = self.cfg.TRAINER.BiMC.PHASE3
        p4 = self.cfg.TRAINER.BiMC.PHASE4
        phase3_enabled = bool(p3.ENABLED)
        phase4_enabled = bool(p4.ENABLED and phase4_diag_vars is not None and p4.COV_MODE != "original")

        s_calib = logits_proto_fused
        hubness = None
        if phase3_enabled and p3.HUBNESS_ENABLED:
            hubness, hubness_stats, hubness_records = self._phase3_hubness(
                source=p3.HUBNESS_SOURCE,
                mixed_proto=fused_proto,
                text_proto=text_proto,
                visual_proto=image_proto,
                tau=p3.HUBNESS_TAU,
                topk=p3.HUBNESS_TOPK,
            )
            s_calib = apply_hubness_correction(s_calib, hubness, lambda_h=p3.HUBNESS_LAMBDA)
            beta_info["phase3"].update(hubness_stats)
            beta_info["phase3"]["hubness_lambda"] = float(p3.HUBNESS_LAMBDA)
            beta_info["phase3"]["hubness_source"] = p3.HUBNESS_SOURCE
            beta_info["hubness_by_class"] = hubness_records

        prob_fused_proto = _score_probabilities(s_calib, "calib")

        logits_cov = _cov_forward(img_feat, image_proto, cov_image)
        logits_knn = knn_similarity_scores(img_feat, description_features, description_targets)    
        s_cov = logits_cov / 512
        s_nn = logits_knn
        prob_cov = _score_probabilities(s_cov, "cov")
        prob_knn = _score_probabilities(s_nn, "nn")

        beta_info["phase3"].update({
            "phase3_enabled": phase3_enabled,
            "temp_scaling_enabled": bool(phase3_enabled and p3.TEMP_SCALING_ENABLED),
            "temp_calib": float(p3.TEMP_CALIB),
            "temp_cov": float(p3.TEMP_COV),
            "temp_nn": float(p3.TEMP_NN),
            "temp_text": float(p3.TEMP_TEXT),
            "temp_visual": float(p3.TEMP_VISUAL),
        })
        if phase3_enabled and p3.ENERGY_ENABLED:
            _record_energy_stats(beta_info, "calib", s_calib)
            _record_energy_stats(beta_info, "cov", s_cov)
            _record_energy_stats(beta_info, "nn", s_nn)

        s_cov4 = None
        prob_cov4 = None
        if phase4_enabled:
            s_cov4 = mahalanobis_diag_score(
                img_feat,
                image_proto,
                phase4_diag_vars,
                temp=p4.COV_SCORE_TEMP,
                eps=p4.COV_EPS,
            )
            prob_cov4 = F.softmax(s_cov4, dim=-1)
            beta_info["phase4"].update(phase3_tensor_stats(s_cov4, prefix="novel_cov_score"))
            beta_info["phase4"].update({
                "phase4_enabled": True,
                "cov_mode": p4.COV_MODE,
                "apply_to_base": bool(p4.APPLY_TO_BASE),
                "apply_to_novel": bool(p4.APPLY_TO_NOVEL),
                "replace_novel_nn": bool(p4.REPLACE_NOVEL_NN),
                "novel_aux_combine_mode": p4.NOVEL_AUX_COMBINE_MODE,
                "cov_score_weight": float(p4.COV_SCORE_WEIGHT),
                "cov_score_temp": float(p4.COV_SCORE_TEMP),
            })
        else:
            beta_info["phase4"]["phase4_enabled"] = False

        NUM_BASE_CLS = num_base_cls
        use_diversity = self.cfg.TRAINER.BiMC.USING_ENSEMBLE
        if use_diversity:
            ensemble_alpha = self.cfg.DATASET.ENSEMBLE_ALPHA
        elif phase4_enabled:
            ensemble_alpha = self.cfg.DATASET.ENSEMBLE_ALPHA
        else:
            ensemble_alpha = 1.0

        base_calib_probs = prob_fused_proto[:, :NUM_BASE_CLS]
        base_aux_probs = prob_cov[:, :NUM_BASE_CLS]
        base_calib_scores = s_calib[:, :NUM_BASE_CLS]
        base_aux_scores = s_cov[:, :NUM_BASE_CLS]

        if phase4_enabled and p4.APPLY_TO_BASE and prob_cov4 is not None:
            cov_weight = _clamped_weight(p4.COV_SCORE_WEIGHT)
            base_aux_probs = (1 - cov_weight) * base_aux_probs + cov_weight * prob_cov4[:, :NUM_BASE_CLS]
            base_aux_scores = (1 - cov_weight) * base_aux_scores + cov_weight * s_cov4[:, :NUM_BASE_CLS]
            beta_info["phase4"]["base_cov4_combined"] = True
        else:
            beta_info["phase4"]["base_cov4_combined"] = False

        inc_calib_probs = prob_fused_proto[:, NUM_BASE_CLS:]
        inc_aux_probs = prob_knn[:, NUM_BASE_CLS:]
        inc_calib_scores = s_calib[:, NUM_BASE_CLS:]
        inc_aux_scores = s_nn[:, NUM_BASE_CLS:]
        inc_aux_temp = p3.TEMP_NN

        if phase4_enabled and p4.APPLY_TO_NOVEL and prob_cov4 is not None and inc_calib_probs.shape[1] > 0:
            inc_aux_probs, inc_aux_scores, inc_aux_temp = self._phase4_novel_aux(
                prob_nn=prob_knn[:, NUM_BASE_CLS:],
                score_nn=s_nn[:, NUM_BASE_CLS:],
                prob_cov4=prob_cov4[:, NUM_BASE_CLS:],
                score_cov4=s_cov4[:, NUM_BASE_CLS:],
                phase3_enabled=phase3_enabled,
                info=beta_info,
            )

        if phase3_enabled and p3.DYNAMIC_ALPHA_ENABLED:
            base_probs = _slice_alpha(
                base_calib_probs,
                base_aux_probs,
                base_calib_scores,
                base_aux_scores,
                "base",
                p3.TEMP_COV,
                beta_info,
            )
            inc_probs = _slice_alpha(
                inc_calib_probs,
                inc_aux_probs,
                inc_calib_scores,
                inc_aux_scores,
                "novel",
                inc_aux_temp,
                beta_info,
            )
        else:
            base_probs = ensemble_alpha * base_calib_probs + (1 - ensemble_alpha) * base_aux_probs
            inc_probs = ensemble_alpha * inc_calib_probs + (1 - ensemble_alpha) * inc_aux_probs

        prob_fused = torch.cat([base_probs, inc_probs], dim=1)
        logits = prob_fused
        if return_beta_info:
            return logits, beta_info
        return logits


    def _phase3_hubness(self, source, mixed_proto, text_proto, visual_proto, tau, topk):
        def _prepare(proto):
            if proto.ndim == 3:
                proto = proto.mean(dim=0)
            return F.normalize(proto, dim=-1)

        mixed = _prepare(mixed_proto)
        text = _prepare(text_proto)
        visual = _prepare(visual_proto)

        if source == "mixed":
            hubness, stats = compute_prototype_hubness(mixed, tau=tau, topk=topk)
        elif source == "text":
            hubness, stats = compute_prototype_hubness(text, tau=tau, topk=topk)
        elif source == "visual":
            hubness, stats = compute_prototype_hubness(visual, tau=tau, topk=topk)
        elif source == "all":
            hubness_parts = []
            part_stats = {}
            for name, proto in (("mixed", mixed), ("text", text), ("visual", visual)):
                part_hubness, stats_i = compute_prototype_hubness(proto, tau=tau, topk=topk)
                hubness_parts.append(part_hubness)
                part_stats[f"{name}_hubness_std"] = stats_i["hubness_std"]
            hubness = torch.stack(hubness_parts, dim=0).mean(dim=0)
            eps = torch.finfo(hubness.dtype).tiny if torch.is_floating_point(hubness) else 1e-8
            hubness = (hubness - hubness.mean()) / hubness.std(unbiased=False).clamp_min(eps)
            hubness = torch.nan_to_num(hubness)
            stats = phase3_tensor_stats(hubness, prefix="hubness")
            stats.update(part_stats)
            stats["hubness_tau"] = float(tau)
            stats["hubness_topk"] = int(topk or 0)
        else:
            raise ValueError(f"Invalid PHASE3.HUBNESS_SOURCE {source}.")

        records = [
            {
                "class_id": int(class_id),
                "hubness": float(value),
                "source": source,
            }
            for class_id, value in enumerate(hubness.detach().float().cpu())
        ]
        return hubness, stats, records


    def _phase4_novel_aux(
        self,
        prob_nn,
        score_nn,
        prob_cov4,
        score_cov4,
        phase3_enabled,
        info,
    ):
        p3 = self.cfg.TRAINER.BiMC.PHASE3
        p4 = self.cfg.TRAINER.BiMC.PHASE4
        mode = p4.NOVEL_AUX_COMBINE_MODE
        info["phase4"]["novel_nn_replaced_or_combined"] = mode

        if p4.REPLACE_NOVEL_NN or mode == "replace":
            return prob_cov4, score_cov4, 1.0
        if mode == "average":
            return 0.5 * prob_nn + 0.5 * prob_cov4, 0.5 * score_nn + 0.5 * score_cov4, 1.0
        if mode == "max":
            return torch.maximum(prob_nn, prob_cov4), torch.maximum(score_nn, score_cov4), 1.0
        if mode == "alpha":
            if phase3_enabled and p3.DYNAMIC_ALPHA_ENABLED:
                alpha_x, alpha_stats = dynamic_alpha_from_scores(
                    score_nn,
                    score_cov4,
                    mode=p3.DYNAMIC_ALPHA_MODE,
                    temp_calib=p3.TEMP_NN if p3.TEMP_SCALING_ENABLED else 1.0,
                    temp_aux=1.0,
                    min_alpha=p3.ALPHA_CLIP_MIN,
                    max_alpha=p3.ALPHA_CLIP_MAX,
                    eps=p3.ALPHA_EPS,
                )
                alpha_stats["split"] = "novel_nn_cov4"
                info["dynamic_alpha"].append(alpha_stats)
                probs = combine_with_dynamic_alpha(prob_nn, prob_cov4, alpha_x)
                scores = alpha_x.unsqueeze(-1) * score_nn + (1 - alpha_x.unsqueeze(-1)) * score_cov4
                return probs, scores, 1.0
            cov_weight = _clamped_weight(p4.COV_SCORE_WEIGHT)
            return (1 - cov_weight) * prob_nn + cov_weight * prob_cov4, (
                (1 - cov_weight) * score_nn + cov_weight * score_cov4
            ), 1.0

        raise ValueError(f"Invalid PHASE4.NOVEL_AUX_COMBINE_MODE {mode}.")



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
