import torch
import torch.nn as nn
import torch.nn.functional as F
import models.clip.clip as clip
import json
from utils.phase1_fusion import compute_query_reliability_beta, fuse_prototypes
from utils.phase2_prototypes import (
    base_neighbor_prior,
    combined_description_reweight,
    compute_visual_quality,
    discriminative_description_reweight,
    dynamic_lambda_i_from_quality,
    normalize as phase2_normalize,
    robust_weighted_visual_prototype,
    shrinkage_visual_prototype,
    visual_grounded_description_reweight,
)
from utils.phase5_space import (
    common_direction_removal,
    diagonal_whitening_apply,
    full_whitening_apply,
    lda_apply,
)
from utils.phase7_separation import (
    graph_highpass_correction,
    hubness_safe_repulsion,
    prototype_repulsion,
)


def _as_float(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().float().cpu().item())
    return float(value)


def _prefixed_stats(stats, prefix):
    return {
        f"{prefix}{key}": value
        for key, value in stats.items()
        if value is not None
    }

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
        self.base_vision_prototype = None
        self.phase2_seen_text_features = None


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
            texts = clip.tokenize(texts).to(self.device)
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
        with open(gpt_path, "r") as file:
            GPT_prompt_dict = json.load(file)
        # The order of embeddings should follow strictly order of classname variable
        # Keys name should match classnames so that we could do fetching from the dict.
        # Convert the dict to lower case
        GPT_prompt_dict = {k.lower().replace("_", " "): v for k, v in GPT_prompt_dict.items()}
        k = cls_begin_index
        for single_key in class_names:
            normalized_key = single_key.lower().replace("_", " ")
            single_class_prompts = GPT_prompt_dict.get(normalized_key)
            if not single_class_prompts:
                single_class_prompts = [self.template[0].format(normalized_key.replace("-", " "))]
            targets = torch.full((len(single_class_prompts),), k)

            k += 1
            x_tokenized = torch.cat([clip.tokenize(p) for p in single_class_prompts])
            with torch.no_grad():
                text_features = self.clip_model.encode_text(x_tokenized.to(self.device))
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_embeddings.append(text_features.mean(0).unsqueeze(0))
            description_embeddings.append(text_features)
            all_targets.append(targets)
        description_embeddings = torch.cat(description_embeddings, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        mean_embeddings = torch.cat(mean_embeddings, dim=0)
        mean_embeddings = F.normalize(mean_embeddings, dim=-1)
        return description_embeddings, all_targets, mean_embeddings


    def soft_calibration(
        self,
        base_protos,
        cur_protos,
        lambda_i=None,
        class_ids=None,
        session_id=None,
        source="fixed",
        return_records=False,
    ):
        shift_weight = self.cfg.TRAINER.BiMC.LAMBDA_I
        tau = self.cfg.TRAINER.BiMC.TAU
        base_protos = F.normalize(base_protos, p=2, dim=-1)
        cur_protos = F.normalize(cur_protos, p=2, dim=-1)
        weights = torch.mm(cur_protos, base_protos.T) * tau
        norm_weights = torch.softmax(weights, dim=1)
        delta_protos = torch.matmul(norm_weights, base_protos)
        delta_protos = F.normalize(delta_protos, p=2, dim=-1)
        if lambda_i is None:
            updated_protos = (1 - shift_weight) * cur_protos + shift_weight * delta_protos
            lambda_values = torch.full(
                (cur_protos.shape[0],),
                float(shift_weight),
                device=cur_protos.device,
                dtype=cur_protos.dtype,
            )
        else:
            lambda_values = lambda_i.to(device=cur_protos.device, dtype=cur_protos.dtype).reshape(-1)
            updated_protos = (1 - lambda_values.unsqueeze(-1)) * cur_protos + lambda_values.unsqueeze(-1) * delta_protos
        updated_protos = F.normalize(updated_protos, dim=-1)
        records = []
        if class_ids is not None:
            for row_id, class_id in enumerate(class_ids):
                records.append({
                    "session": int(session_id) if session_id is not None else None,
                    "class_id": int(class_id),
                    "lambda_i": _as_float(lambda_values[row_id]),
                    "lambda_source": source,
                })
        if return_records:
            return updated_protos, records
        return updated_protos


    def calibrated_text_proto(self, text_features, description_proto):
        if self.cfg.TRAINER.BiMC.TEXT_CALIBRATION:
            lambda_t = self.cfg.TRAINER.BiMC.LAMBDA_T
        else:
            lambda_t = 0.0
        return (1 - lambda_t) * text_features + lambda_t * description_proto
    

    def build_task_statistics(self, class_names, loader,
                         class_index, calibrate_novel_vision_proto=False, task_id=None):
        
            
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
        session_id = int(task_id) if task_id is not None else self._session_id_from_class_index(class_index)
        if cls_begin_index == 0:
            self.phase2_seen_text_features = None


        text_features, text_targets = self.inference_text_feature(class_names, self.template, cls_begin_index)

        description_features, description_targets, description_proto = \
                                  self.inference_all_description_feature(class_names=class_names, 
                                  gpt_path=self.cfg.DATASET.GPT_PATH,
                                  cls_begin_index=cls_begin_index)
        
        images_features, images_targets, images_proto = \
                                    self.inference_all_img_feature(loader, cls_begin_index)
        if self.phase2_seen_text_features is None:
            all_class_name_protos = text_features
        else:
            all_class_name_protos = torch.cat([self.phase2_seen_text_features, text_features], dim=0)

        images_proto, support_visual_proto, phase2_records, visual_quality = self._apply_phase2_visual_prototypes(
            images_features=images_features,
            images_targets=images_targets,
            images_proto=images_proto,
            text_features=text_features,
            class_index=class_index,
            session_id=session_id,
        )

        description_proto, text_records = self._apply_phase2_text_prototypes(
            description_features=description_features,
            description_targets=description_targets,
            description_proto=description_proto,
            text_features=text_features,
            all_class_name_protos=all_class_name_protos,
            support_visual_proto=support_visual_proto,
            class_index=class_index,
            session_id=session_id,
        )
        self._merge_phase2_records(phase2_records, text_records)
        self.phase2_seen_text_features = all_class_name_protos.detach()

        if cls_begin_index != 0:
            if calibrate_novel_vision_proto:
                print(f'calibrate vision proto on class [{class_index}]')
                lambda_i = None
                lambda_source = "fixed"
                p2_opts = self.cfg.TRAINER.BiMC.PHASE2
                if p2_opts.ENABLED and p2_opts.DYNAMIC_LAMBDA_I:
                    lambda_i = torch.stack([
                        dynamic_lambda_i_from_quality(
                            visual_quality[int(class_id)],
                            min_val=p2_opts.DYNAMIC_LAMBDA_MIN,
                            max_val=p2_opts.DYNAMIC_LAMBDA_MAX,
                        )
                        for class_id in class_index
                    ])
                    lambda_source = "dynamic_quality"
                images_proto, calibration_records = self.soft_calibration(
                    self.base_vision_prototype,
                    images_proto,
                    lambda_i=lambda_i,
                    class_ids=class_index,
                    session_id=session_id,
                    source=lambda_source,
                    return_records=True,
                )
                self._merge_phase2_records(phase2_records, calibration_records)
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
            'phase2_records': phase2_records,
            
            'class_index': class_index,
            'sample_cnt': len(images_features)
        }


    def _apply_phase2_visual_prototypes(
        self,
        images_features,
        images_targets,
        images_proto,
        text_features,
        class_index,
        session_id,
    ):
        p2_opts = self.cfg.TRAINER.BiMC.PHASE2
        records = []
        visual_quality = {}
        mode = p2_opts.VISUAL_PROTO_MODE if p2_opts.ENABLED else "mean"
        class_ids = [int(class_id) for class_id in class_index]

        for rel_idx, class_id in enumerate(class_ids):
            class_mask = images_targets == class_id
            class_features = images_features[class_mask]
            quality = compute_visual_quality(class_features)
            visual_quality[class_id] = quality
            records.append({
                "session": int(session_id),
                "class_id": class_id,
                "visual_proto_mode": mode,
                "visual_quality": _as_float(quality),
                "visual_fallback": "",
                "num_support": int(class_features.shape[0]),
            })

        if not p2_opts.ENABLED or mode == "mean":
            return images_proto, images_proto, records, visual_quality

        new_protos = []
        for rel_idx, class_id in enumerate(class_ids):
            record = records[rel_idx]
            class_features = images_features[images_targets == class_id]
            shot_proto = images_proto[rel_idx]
            new_proto = shot_proto

            if int(class_index[0]) == 0:
                record["visual_fallback"] = "base_session_mean"

            elif mode == "robust_weighted":
                new_proto, _, stats = robust_weighted_visual_prototype(
                    class_features,
                    kappa=p2_opts.ROBUST_KAPPA,
                    drop_lowest=p2_opts.ROBUST_DROP_LOWEST,
                )
                record.update(_prefixed_stats(stats, prefix="visual_"))

            elif mode == "shrinkage_base_prior":
                prior_proto, fallback = self._phase2_visual_prior(
                    shot_proto=shot_proto,
                    text_proto=text_features[rel_idx],
                )
                if fallback:
                    record["visual_fallback"] = fallback
                else:
                    new_proto, rho, stats = shrinkage_visual_prototype(
                        class_features,
                        shot_proto,
                        prior_proto,
                        eps=p2_opts.SHRINKAGE_EPS,
                        min_rho=p2_opts.SHRINKAGE_MIN,
                        max_rho=p2_opts.SHRINKAGE_MAX,
                    )
                    record["shrinkage_rho"] = _as_float(rho)
                    record.update(_prefixed_stats(stats, prefix="shrinkage_"))

            new_protos.append(new_proto)

        return torch.stack(new_protos, dim=0), torch.stack(new_protos, dim=0), records, visual_quality


    def _phase2_visual_prior(self, shot_proto, text_proto):
        p2_opts = self.cfg.TRAINER.BiMC.PHASE2
        if p2_opts.SHRINKAGE_PRIOR == "base_neighbors":
            if self.base_vision_prototype is None:
                return shot_proto, "missing_base_prototypes"
            return base_neighbor_prior(shot_proto, self.base_vision_prototype, tau=self.cfg.TRAINER.BiMC.TAU), ""
        if p2_opts.SHRINKAGE_PRIOR == "text":
            return phase2_normalize(text_proto, dim=-1), ""
        if p2_opts.SHRINKAGE_PRIOR == "zero":
            return torch.zeros_like(shot_proto), ""
        return shot_proto, "invalid_prior"


    def _apply_phase2_text_prototypes(
        self,
        description_features,
        description_targets,
        description_proto,
        text_features,
        all_class_name_protos,
        support_visual_proto,
        class_index,
        session_id,
    ):
        p2_opts = self.cfg.TRAINER.BiMC.PHASE2
        mode = p2_opts.TEXT_PROTO_MODE if p2_opts.ENABLED else "mean"
        records = []

        if not p2_opts.ENABLED or mode == "mean":
            for rel_idx, class_id in enumerate(class_index):
                desc = description_features[description_targets.to(description_features.device) == int(class_id)]
                records.append(self._mean_description_record(desc, int(session_id), int(class_id), mode))
            return description_proto, records

        new_description_proto = []
        targets = description_targets.to(description_features.device)
        for rel_idx, class_id in enumerate(class_index):
            class_id = int(class_id)
            desc = description_features[targets == class_id]
            record = {
                "session": int(session_id),
                "class_id": class_id,
                "text_proto_mode": mode,
                "text_fallback": "",
                "num_descriptions": int(desc.shape[0]),
            }

            if desc.shape[0] == 0:
                new_description_proto.append(description_proto[rel_idx])
                record["text_fallback"] = "missing_descriptions"
                records.append(record)
                continue

            if mode == "discriminative_reweight":
                proto, _, stats = discriminative_description_reweight(
                    desc,
                    text_features[rel_idx],
                    all_class_name_protos,
                    temp=p2_opts.DESC_TEMP,
                    topk=None,
                )
            elif mode == "visual_grounded_reweight":
                proto, _, stats = visual_grounded_description_reweight(
                    desc,
                    support_visual_proto[rel_idx],
                    temp=p2_opts.DESC_TEMP,
                )
            elif mode == "combined_reweight":
                proto, _, stats = combined_description_reweight(
                    desc,
                    support_visual_proto[rel_idx],
                    text_features[rel_idx],
                    all_class_name_protos,
                    temp=p2_opts.DESC_TEMP,
                    lambda_d=p2_opts.DESC_LAMBDA_D,
                )
            elif mode == "topk_discriminative":
                proto, _, stats = discriminative_description_reweight(
                    desc,
                    text_features[rel_idx],
                    all_class_name_protos,
                    temp=p2_opts.DESC_TEMP,
                    topk=p2_opts.DESC_TOPK,
                )
            else:
                proto = description_proto[rel_idx]
                stats = {}
                record["text_fallback"] = "invalid_text_mode"

            new_description_proto.append(proto)
            record.update(stats)
            records.append(record)

        return torch.stack(new_description_proto, dim=0), records


    def _mean_description_record(self, desc, session_id, class_id, mode):
        if desc.shape[0] == 0:
            entropy = None
        else:
            entropy = float(torch.log(torch.tensor(float(desc.shape[0]))).item())
        return {
            "session": int(session_id),
            "class_id": int(class_id),
            "text_proto_mode": mode,
            "text_fallback": "",
            "num_descriptions": int(desc.shape[0]),
            "desc_weight_entropy": entropy,
            "desc_weight_min": float(1.0 / desc.shape[0]) if desc.shape[0] else None,
            "desc_weight_max": float(1.0 / desc.shape[0]) if desc.shape[0] else None,
        }


    def _merge_phase2_records(self, target_records, incoming_records):
        by_key = {
            (record.get("session"), record.get("class_id")): record
            for record in target_records
        }
        for record in incoming_records:
            key = (record.get("session"), record.get("class_id"))
            if key in by_key:
                by_key[key].update(record)
            else:
                target_records.append(record)
                by_key[key] = record


    def _session_id_from_class_index(self, class_index):
        first_class = int(class_index[0])
        if first_class == 0:
            return 0
        inc = max(1, int(self.cfg.DATASET.NUM_INC_CLS))
        return 1 + max(0, first_class - int(self.cfg.DATASET.NUM_INIT_CLS)) // inc

   

    def phase5_transform_tensor(self, x, phase5_state):
        if not phase5_state or not phase5_state.get("enabled", False):
            return x

        mode = phase5_state.get("mode", "none")
        try:
            if mode == "common_direction_removal":
                return common_direction_removal(
                    x,
                    phase5_state["direction"],
                    rho=phase5_state.get("rho", 0.5),
                )
            if mode == "whitening":
                transform = phase5_state["transform"]
                if transform.get("type") == "full_whitening":
                    return full_whitening_apply(x, transform)
                return diagonal_whitening_apply(x, transform)
            if mode == "lda_shrinkage":
                return lda_apply(x, phase5_state["projection"])
        except (RuntimeError, ValueError, KeyError):
            return x
        return x


    def phase7_transform_prototypes(self, prototypes, phase7_state):
        if not phase7_state or not phase7_state.get("enabled", False):
            return prototypes
        if phase7_state.get("fallback_reason"):
            return prototypes

        if prototypes.ndim == 3:
            return torch.stack(
                [self.phase7_transform_prototypes(item, phase7_state) for item in prototypes],
                dim=0,
            )

        mode = phase7_state.get("mode", "none")
        try:
            if mode == "prototype_repulsion":
                updated, _ = prototype_repulsion(
                    prototypes,
                    delta=phase7_state.get("delta", 0.03),
                    margin=phase7_state.get("margin", 0.0),
                    topk=phase7_state.get("topk", 5),
                )
                return updated
            if mode == "graph_highpass":
                updated, _ = graph_highpass_correction(
                    prototypes,
                    tau=phase7_state.get("tau", 0.05),
                    gamma=phase7_state.get("gamma", 0.03),
                    topk=phase7_state.get("topk", 5),
                    normalize_adj=phase7_state.get("normalize_adj", True),
                )
                return updated
            if mode == "hubness_safe_repulsion":
                hubness = phase7_state.get("hubness")
                if hubness is None:
                    return prototypes
                updated, _ = hubness_safe_repulsion(
                    prototypes,
                    hubness=hubness,
                    delta=phase7_state.get("delta", 0.03),
                    topk=phase7_state.get("topk", 5),
                )
                return updated
        except (RuntimeError, ValueError, KeyError):
            return prototypes
        return prototypes


    def forward_ours(self, images, num_cls, num_base_cls,
                           image_proto, cov_image,
                           description_proto,
                           description_features, description_targets,
                           text_features,
                           beta,
                           return_beta_info=False,
                           phase5_state=None,
                           phase7_state=None):
    
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
        img_feat_for_proto = img_feat
        text_proto_for_fusion = text_proto
        image_proto_for_fusion = image_proto

        if phase5_state and phase5_state.get("enabled", False):
            apply_to = phase5_state.get("apply_to", "all")
            if apply_to in ("query_only", "all"):
                img_feat_for_proto = self.phase5_transform_tensor(img_feat_for_proto, phase5_state)
            if apply_to in ("prototype_only", "all"):
                text_proto_for_fusion = self.phase5_transform_tensor(text_proto_for_fusion, phase5_state)
                image_proto_for_fusion = self.phase5_transform_tensor(image_proto_for_fusion, phase5_state)

        if phase7_state and phase7_state.get("enabled", False):
            source = phase7_state.get("source", "mixed")
            if source == "text":
                text_proto_for_fusion = self.phase7_transform_prototypes(text_proto_for_fusion, phase7_state)
            elif source == "visual":
                image_proto_for_fusion = self.phase7_transform_prototypes(image_proto_for_fusion, phase7_state)

        phase1_opts = self.cfg.TRAINER.BiMC
        beta_info = {
            "mode": phase1_opts.FUSION_BETA_MODE,
            "geometry": phase1_opts.FUSION_GEOMETRY,
            "beta": None,
        }

        # Preserve the original BiMC formula exactly for the default baseline.
        if phase1_opts.FUSION_BETA_MODE == "fixed" and phase1_opts.FUSION_GEOMETRY == "linear":
            fused_proto = beta * text_proto_for_fusion + (1 - beta) * image_proto_for_fusion
            fused_proto = F.normalize(fused_proto, dim=-1)
            if phase7_state and phase7_state.get("enabled", False) and phase7_state.get("source", "mixed") == "mixed":
                fused_proto = self.phase7_transform_prototypes(fused_proto, phase7_state)
            logits_proto_fused = img_feat_for_proto @ fused_proto.t()
        else:
            text_proto_for_fusion = F.normalize(text_proto_for_fusion, dim=-1)
            image_proto_for_fusion = F.normalize(image_proto_for_fusion, dim=-1)

            if phase1_opts.FUSION_BETA_MODE == "query_reliability":
                beta_x = compute_query_reliability_beta(
                    img_feat_for_proto,
                    text_proto_for_fusion,
                    image_proto_for_fusion,
                    reliability_mode=phase1_opts.RELIABILITY_MODE,
                    beta_clip_min=phase1_opts.BETA_CLIP_MIN,
                    beta_clip_max=phase1_opts.BETA_CLIP_MAX,
                )
                fused_proto = fuse_prototypes(
                    image_proto_for_fusion.unsqueeze(0),
                    text_proto_for_fusion.unsqueeze(0),
                    beta_x.view(-1, 1, 1),
                    geometry=phase1_opts.FUSION_GEOMETRY,
                )
                if phase7_state and phase7_state.get("enabled", False) and phase7_state.get("source", "mixed") == "mixed":
                    fused_proto = self.phase7_transform_prototypes(fused_proto, phase7_state)
                logits_proto_fused = torch.einsum("bd,bcd->bc", img_feat_for_proto, fused_proto)
                beta_info["beta"] = beta_x.detach()
            else:
                fused_proto = fuse_prototypes(
                    image_proto_for_fusion,
                    text_proto_for_fusion,
                    beta,
                    geometry=phase1_opts.FUSION_GEOMETRY,
                )
                if phase7_state and phase7_state.get("enabled", False) and phase7_state.get("source", "mixed") == "mixed":
                    fused_proto = self.phase7_transform_prototypes(fused_proto, phase7_state)
                logits_proto_fused = img_feat_for_proto @ fused_proto.t()
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
