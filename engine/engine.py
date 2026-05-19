import torch
import torch.nn as nn
from datasets.data_manager import DatasetManager
from tqdm import tqdm
from utils.evaluator import AccuracyEvaluator
from utils.phase1_fusion import beta_statistics, compute_class_margin_beta, validate_phase1_options
from utils.phase1_report import write_phase1_outputs
from utils.phase2_prototypes import validate_phase2_options
from utils.phase5_space import (
    diagonal_whitening_fit,
    estimate_common_direction,
    full_whitening_fit,
    lda_shrinkage_fit,
    validate_phase5_options,
)
from utils.phase6_alignment import (
    apply_label_prior_correction,
    blackbox_shift_prior,
    label_prior_stats,
    ot_text_image_alignment,
    prediction_frequency_prior,
    support_aware_text_proto,
    validate_phase6_options,
)
from utils.phase7_separation import (
    graph_highpass_correction,
    hubness_safe_repulsion,
    prototype_repulsion,
    validate_phase7_options,
)
from models.bimc import BiMC
import numpy as np
import time


class Runner:

    def __init__(self, cfg):
        self.cfg = cfg
        validate_phase1_options(cfg)
        validate_phase2_options(cfg)
        validate_phase5_options(cfg)
        validate_phase6_options(cfg)
        validate_phase7_options(cfg)
        self.data_manager = DatasetManager(cfg,) 
        self.device = cfg.DEVICE.DEVICE_NAME

        self.model = BiMC(cfg, self.data_manager.template, self.device)

        # device
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
            self.is_distributed = True
        else:
            self.is_distributed = False
            

        self.acc_list = []
        self.task_acc_list = []
        self.eval_results = []
        self.beta_session_records = []
        self.beta_class_records = []
        self.phase2_records = []
        self.phase5_records = []
        self.phase6_records = []
        self.phase7_records = []
        self.label_prior_records = []
        self.evaluator = AccuracyEvaluator(self.data_manager.class_index_in_task)


    def _model_impl(self):
        if self.is_distributed:
            return self.model.module
        return self.model


    def merge_dicts(self, dict_list):
        result = {}

        keys_to_merge = [
            'description_proto', 
            'description_features', 
            'description_targets',
            'text_features', 
            'text_targets',
            'image_proto', 
            'images_features', 
            'images_targets'
        ]

        for key in keys_to_merge:
            result[key] = torch.cat([d[key] for d in dict_list], dim=0)


        weights = [len(d['class_index']) for d in dict_list]


        cov_keys = [
            'cov_image',
        ]
        cov_sums = {key: torch.zeros_like(dict_list[0][key]) for key in cov_keys}
        weight_sum = sum(weights)

        for i, d in enumerate(dict_list):
            for key in cov_keys:
                cov_sums[key] += d[key] * weights[i]

        for key in cov_keys:
            if weight_sum > 0: 
                result[key] = cov_sums[key] / weight_sum

        return result



    @torch.no_grad()
    def run(self):
        print(f'Start inferencing on all tasks: [0, {self.data_manager.num_tasks - 1}]')
        state_dict_list = []
        for i in range(self.data_manager.num_tasks):
            self.model.eval()

            current_class_name = np.array(self.data_manager.class_names)[self.data_manager.class_index_in_task[i]]
            loader = self.data_manager.get_dataloader(i, source='train', mode='test', accumulate_past=False)            



            current_state_dict = self._model_impl().build_task_statistics(current_class_name, loader,
                                                             class_index=self.data_manager.class_index_in_task[i], 
                                                             calibrate_novel_vision_proto=self.cfg.TRAINER.BiMC.VISION_CALIBRATION,
                                                             task_id=i,)

            state_dict_list.append(current_state_dict)            
            self.phase2_records.extend(current_state_dict.get("phase2_records", []))
            merged_state_dict = self.merge_dicts(state_dict_list)

            start_time = time.time()
            acc = self.inference_task_covariance(i, merged_state_dict)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f'+++++++++++  task {i}, time: {elapsed_time} ++++++++++++++++')

            print(f'=> Task [{i}], Acc: {acc["mean_acc"]:.3f}')
            self.acc_list.append(round(acc["mean_acc"], 3))
            self.task_acc_list.append(acc['task_acc'])
            self.eval_results.append(acc)

        print(f'Final acc:{self.acc_list}')
        print('Task-wise acc:')
        for i, task_acc in enumerate(self.task_acc_list):
            print(f'task {i:2d}, acc:{task_acc}')
        return self._save_phase1_outputs()
    

    @torch.no_grad()
    def inference_task_covariance(self, task_id, state_dict):

        beta, beta_values, beta_class_records = self._prepare_session_beta(task_id, state_dict)
        state_dict = dict(state_dict)
        phase6_records = self._apply_phase6_alignment(task_id, state_dict)
        self.phase6_records.extend(phase6_records)
        phase5_state, phase5_record = self._prepare_phase5_state(task_id, state_dict)
        self.phase5_records.append(phase5_record)
        phase7_state, phase7_record = self._prepare_phase7_state(task_id, state_dict, phase5_state, beta)
        self.phase7_records.append(phase7_record)

        image_proto = state_dict['image_proto']
        cov_image = state_dict['cov_image']
        text_features = state_dict['text_features']
        description_proto = state_dict['description_proto']
        description_features = state_dict['description_features']
        description_targets = state_dict['description_targets']

        num_base_class = len(self.data_manager.class_index_in_task[0])
        num_accumulated_class = max(self.data_manager.class_index_in_task[task_id]) + 1
        
        test_loader = self.data_manager.get_dataloader(task_id, source='test', mode='test')
        all_logits = []
        all_targets = []
        beta_chunks = []
        if beta_values is not None:
            beta_chunks.append(beta_values)

        for i, batch in enumerate(tqdm(test_loader)):
            data, targets = self.parse_batch(batch)
            logits, beta_info = self._model_impl().forward_ours(data, num_accumulated_class, num_base_class,
                                                   image_proto, 
                                                   cov_image,
                                                   description_proto,
                                                   description_features, 
                                                   description_targets,
                                                   text_features,
                                                   beta=beta,
                                                   return_beta_info=True,
                                                   phase5_state=phase5_state,
                                                   phase7_state=phase7_state)
            if beta_info.get("beta") is not None:
                beta_chunks.append(beta_info["beta"])

            all_logits.append(logits)
            all_targets.append(targets)

        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_logits, prior_record = self._apply_phase6_label_prior(task_id, all_logits)
        if prior_record is not None:
            self.label_prior_records.append(prior_record)

        eval_acc = self.evaluator.calc_accuracy(all_logits, all_targets, task_id) 
        beta_record = {"session": int(task_id)}
        beta_record.update(beta_statistics(beta_chunks))
        self._assert_beta_range(beta_chunks)
        eval_acc["beta_stats"] = beta_record
        self.beta_session_records.append(beta_record)
        self.beta_class_records.extend(beta_class_records)
        print(f"Test acc mean: {eval_acc['mean_acc']}, task-wise acc: {eval_acc['task_acc']}")
        return eval_acc


    def _prepare_session_beta(self, task_id, state_dict):
        opts = self.cfg.TRAINER.BiMC
        num_accumulated_class = max(self.data_manager.class_index_in_task[task_id]) + 1

        if opts.FUSION_BETA_MODE == "fixed":
            beta = float(self.cfg.DATASET.BETA)
            beta_values = torch.full((num_accumulated_class,), beta)
            return beta, beta_values, []

        if opts.FUSION_BETA_MODE == "query_reliability":
            return float(self.cfg.DATASET.BETA), None, []

        text_proto = self._model_impl().calibrated_text_proto(
            state_dict["text_features"],
            state_dict["description_proto"],
        )
        beta = compute_class_margin_beta(
            support_features=state_dict["images_features"],
            support_labels=state_dict["images_targets"],
            text_proto=text_proto,
            visual_proto=state_dict["image_proto"],
            beta_temperature=opts.BETA_TEMPERATURE,
            beta_clip_min=opts.BETA_CLIP_MIN,
            beta_clip_max=opts.BETA_CLIP_MAX,
            default_beta=self.cfg.DATASET.BETA,
            use_leave_one_out=True,
        )
        beta_class_records = [
            {
                "session": int(task_id),
                "class_id": int(class_id),
                "beta": float(value),
            }
            for class_id, value in enumerate(beta.detach().cpu())
        ]
        return beta, beta.detach().cpu(), beta_class_records


    def _assert_beta_range(self, beta_chunks):
        tensors = []
        for chunk in beta_chunks:
            if chunk is None:
                continue
            if isinstance(chunk, torch.Tensor):
                tensors.append(chunk.detach().reshape(-1).cpu())
            else:
                tensors.append(torch.as_tensor(chunk).reshape(-1).cpu())
        if not tensors:
            return
        values = torch.cat(tensors, dim=0)
        min_beta = self.cfg.TRAINER.BiMC.BETA_CLIP_MIN
        max_beta = self.cfg.TRAINER.BiMC.BETA_CLIP_MAX
        if values.min().item() < min_beta - 1e-6 or values.max().item() > max_beta + 1e-6:
            raise ValueError(f"Beta values outside [{min_beta}, {max_beta}].")


    def _apply_phase6_alignment(self, task_id, state_dict):
        opts = self.cfg.TRAINER.BiMC.PHASE6
        mode = opts.ALIGNMENT_MODE if opts.ENABLED else "none"
        if mode in ("none", "label_prior_correction"):
            return []

        description_features = state_dict["description_features"]
        description_targets = state_dict["description_targets"].to(description_features.device)
        image_features = state_dict["images_features"]
        image_targets = state_dict["images_targets"].to(image_features.device)
        image_proto = state_dict["image_proto"]
        old_description_proto = state_dict["description_proto"]

        updated_proto = []
        records = []
        for class_id in range(old_description_proto.shape[0]):
            desc = description_features[description_targets == class_id]
            support = image_features[image_targets == class_id]
            fallback_reason = ""
            if mode == "ot_text_image":
                if opts.OT_USE_SINKHORN:
                    proto, _, stats = ot_text_image_alignment(
                        desc,
                        support,
                        eps=opts.OT_EPS,
                        max_iter=opts.OT_MAX_ITER,
                        normalize_cost=opts.OT_NORMALIZE_COST,
                    )
                else:
                    support_proto = image_proto[class_id]
                    proto, _, stats = support_aware_text_proto(
                        desc,
                        support_proto,
                        temp=opts.SUPPORT_AWARE_TEXT_TEMP,
                        topk=opts.SUPPORT_AWARE_TEXT_TOPK,
                    )
                    fallback_reason = "sinkhorn_disabled"
            elif mode == "support_aware_text":
                support_proto = image_proto[class_id]
                proto, _, stats = support_aware_text_proto(
                    desc,
                    support_proto,
                    temp=opts.SUPPORT_AWARE_TEXT_TEMP,
                    topk=opts.SUPPORT_AWARE_TEXT_TOPK,
                )
            else:
                proto = old_description_proto[class_id]
                stats = {"fallback_reason": "invalid_alignment_mode"}

            if stats.get("fallback_reason"):
                proto = old_description_proto[class_id]
            updated_proto.append(proto.to(device=old_description_proto.device, dtype=old_description_proto.dtype))

            record = {
                "session": int(task_id),
                "class_id": int(class_id),
                "phase6_enabled": bool(opts.ENABLED),
                "alignment_mode": mode,
                "num_descriptions": int(desc.shape[0]),
                "num_support": int(support.shape[0]),
            }
            record.update(stats)
            if fallback_reason:
                record["fallback_reason"] = fallback_reason
            records.append(record)

        state_dict["description_proto"] = torch.stack(updated_proto, dim=0)
        return records


    def _prepare_phase5_state(self, task_id, state_dict):
        opts = self.cfg.TRAINER.BiMC.PHASE5
        mode = opts.SPACE_TRANSFORM if opts.ENABLED else "none"
        record = {
            "session": int(task_id),
            "phase5_enabled": bool(opts.ENABLED),
            "space_transform": mode,
            "apply_to": opts.APPLY_TO,
            "fallback_reason": "",
        }
        state = {"enabled": False}
        if not opts.ENABLED or mode == "none":
            return state, record

        try:
            if mode == "common_direction_removal":
                prototypes = self._phase5_prototype_source(state_dict)
                features = self._phase5_feature_source(
                    state_dict,
                    opts.CDR_SOURCE if opts.CDR_SOURCE != "prototypes" else "all_seen_features",
                )
                direction, stats = estimate_common_direction(
                    prototypes=prototypes,
                    features=features,
                    source=opts.CDR_SOURCE,
                )
                state = {
                    "enabled": not bool(stats.get("fallback_reason")),
                    "mode": mode,
                    "apply_to": opts.APPLY_TO,
                    "direction": direction,
                    "rho": opts.CDR_RHO,
                    "fallback_reason": stats.get("fallback_reason", ""),
                }
                record.update(stats)
                record["cdr_rho"] = float(opts.CDR_RHO)

            elif mode == "whitening":
                features = self._phase5_feature_source(state_dict, opts.WHITENING_SOURCE)
                transform = (
                    diagonal_whitening_fit(features, eps=opts.WHITENING_EPS)
                    if opts.WHITENING_DIAG_ONLY
                    else full_whitening_fit(features, eps=opts.WHITENING_EPS)
                )
                state = {
                    "enabled": True,
                    "mode": mode,
                    "apply_to": opts.APPLY_TO,
                    "transform": transform,
                    "fallback_reason": transform["stats"].get("fallback_reason", ""),
                }
                record.update(transform["stats"])
                record["whitening_source"] = opts.WHITENING_SOURCE
                record["whitening_diag_only"] = bool(opts.WHITENING_DIAG_ONLY)

            elif mode == "lda_shrinkage":
                if opts.APPLY_TO != "all":
                    raise ValueError("lda_shrinkage_requires_apply_to_all")
                base_by_class, support_by_class = self._phase5_lda_sources(state_dict)
                transform = lda_shrinkage_fit(
                    base_by_class,
                    support_features_by_class=support_by_class if opts.LDA_SOURCE == "base_plus_support" else None,
                    dim=opts.LDA_DIM,
                    gamma=opts.LDA_GAMMA if opts.LDA_USE_SHRINKAGE else 0.0,
                    novel_weight=opts.LDA_NOVEL_WEIGHT,
                )
                state = {
                    "enabled": True,
                    "mode": mode,
                    "apply_to": opts.APPLY_TO,
                    "projection": transform,
                    "fallback_reason": transform["stats"].get("fallback_reason", ""),
                }
                record.update(transform["stats"])
                record["lda_source"] = opts.LDA_SOURCE
                record["lda_use_shrinkage"] = bool(opts.LDA_USE_SHRINKAGE)

        except (RuntimeError, ValueError, KeyError) as exc:
            record["fallback_reason"] = str(exc)
            state = {
                "enabled": False,
                "mode": mode,
                "apply_to": opts.APPLY_TO,
                "fallback_reason": str(exc),
            }

        return state, record


    def _prepare_phase7_state(self, task_id, state_dict, phase5_state, beta):
        opts = self.cfg.TRAINER.BiMC.PHASE7
        mode = opts.SEPARATION_MODE if opts.ENABLED else "none"
        record = {
            "session": int(task_id),
            "phase7_enabled": bool(opts.ENABLED),
            "separation_mode": mode,
            "separation_source": opts.REPULSION_SOURCE,
            "fallback_reason": "",
        }
        state = {"enabled": False}
        if not opts.ENABLED or mode == "none":
            return state, record

        state = {
            "enabled": True,
            "mode": mode,
            "source": opts.REPULSION_SOURCE,
            "delta": opts.REPULSION_DELTA,
            "margin": opts.REPULSION_MARGIN,
            "topk": opts.REPULSION_TOPK if mode != "graph_highpass" else opts.GRAPH_TOPK,
            "tau": opts.GRAPH_TAU,
            "gamma": opts.GRAPH_GAMMA,
            "normalize_adj": bool(opts.GRAPH_NORMALIZE_ADJ),
            "fallback_reason": "",
        }

        try:
            prototypes = self._phase7_source_prototypes(state_dict, phase5_state, beta, opts.REPULSION_SOURCE)
            hubness = self._phase7_hubness(state_dict, phase5_state, prototypes)
            state["hubness"] = hubness
            if mode == "prototype_repulsion":
                _, stats = prototype_repulsion(
                    prototypes,
                    delta=opts.REPULSION_DELTA,
                    margin=opts.REPULSION_MARGIN,
                    topk=opts.REPULSION_TOPK,
                )
            elif mode == "graph_highpass":
                _, stats = graph_highpass_correction(
                    prototypes,
                    tau=opts.GRAPH_TAU,
                    gamma=opts.GRAPH_GAMMA,
                    topk=opts.GRAPH_TOPK,
                    normalize_adj=opts.GRAPH_NORMALIZE_ADJ,
                )
            elif mode == "hubness_safe_repulsion":
                _, stats = hubness_safe_repulsion(
                    prototypes,
                    hubness=hubness,
                    delta=opts.REPULSION_DELTA,
                    topk=opts.REPULSION_TOPK,
                )
            else:
                stats = {"fallback_reason": "invalid_separation_mode"}
            record.update(stats)
            state["fallback_reason"] = stats.get("fallback_reason", "")
            state["enabled"] = not bool(state["fallback_reason"])
        except (RuntimeError, ValueError, KeyError) as exc:
            record["fallback_reason"] = str(exc)
            state["enabled"] = False
            state["fallback_reason"] = str(exc)
        return state, record


    def _apply_phase6_label_prior(self, task_id, logits):
        opts = self.cfg.TRAINER.BiMC.PHASE6
        if not opts.ENABLED:
            return logits, None
        if not opts.LABEL_PRIOR_ENABLED and opts.ALIGNMENT_MODE != "label_prior_correction":
            return logits, None

        probs = logits / logits.sum(dim=1, keepdim=True).clamp_min(opts.LABEL_PRIOR_EPS)
        mode = opts.LABEL_PRIOR_MODE
        if mode == "prediction_frequency":
            prior = prediction_frequency_prior(probs, eps=opts.LABEL_PRIOR_EPS)
        elif mode == "uniform_smoothing":
            freq = prediction_frequency_prior(probs, eps=opts.LABEL_PRIOR_EPS)
            uniform = torch.full_like(freq, 1.0 / max(1, freq.numel()))
            prior = 0.5 * freq + 0.5 * uniform
            prior = prior / prior.sum().clamp_min(opts.LABEL_PRIOR_EPS)
        else:
            prior = blackbox_shift_prior(
                probs,
                max_iter=opts.LABEL_PRIOR_MAX_ITER,
                eps=opts.LABEL_PRIOR_EPS,
            )

        scores = torch.log(probs.clamp_min(opts.LABEL_PRIOR_EPS))
        corrected = apply_label_prior_correction(
            scores,
            prior,
            strength=opts.LABEL_PRIOR_STRENGTH,
            eps=opts.LABEL_PRIOR_EPS,
        )
        record = {
            "session": int(task_id),
            "phase6_enabled": bool(opts.ENABLED),
            "alignment_mode": opts.ALIGNMENT_MODE,
        }
        record.update(label_prior_stats(
            prior,
            mode=mode,
            strength=opts.LABEL_PRIOR_STRENGTH,
            transductive=bool(opts.LABEL_PRIOR_TRANSDUCTIVE),
            level="session",
        ))
        return corrected, record


    def _phase5_prototype_source(self, state_dict):
        text_proto = self._model_impl().calibrated_text_proto(
            state_dict["text_features"],
            state_dict["description_proto"],
        )
        return torch.cat([state_dict["image_proto"], text_proto], dim=0)


    def _phase5_feature_source(self, state_dict, source):
        features = state_dict["images_features"]
        labels = state_dict["images_targets"].to(features.device)
        if source == "base_features":
            num_base = len(self.data_manager.class_index_in_task[0])
            mask = labels < num_base
            selected = features[mask]
            if selected.numel() == 0:
                raise ValueError("missing_base_features")
            return selected
        if source == "all_seen_features":
            if features.numel() == 0:
                raise ValueError("missing_all_seen_features")
            return features
        raise ValueError(f"invalid_feature_source:{source}")


    def _phase5_lda_sources(self, state_dict):
        features = state_dict["images_features"]
        labels = state_dict["images_targets"].to(features.device)
        num_base = len(self.data_manager.class_index_in_task[0])
        base_by_class = {}
        support_by_class = {}
        for class_id in range(state_dict["image_proto"].shape[0]):
            class_features = features[labels == class_id]
            if class_features.numel() == 0:
                continue
            if class_id < num_base:
                base_by_class[class_id] = class_features
            else:
                support_by_class[class_id] = class_features
        if len(base_by_class) < 2:
            raise ValueError("missing_base_class_features_for_lda")
        return base_by_class, support_by_class


    def _phase7_source_prototypes(self, state_dict, phase5_state, beta, source):
        text_proto = self._model_impl().calibrated_text_proto(
            state_dict["text_features"],
            state_dict["description_proto"],
        )
        image_proto = state_dict["image_proto"]
        if phase5_state and phase5_state.get("enabled") and phase5_state.get("apply_to") in ("prototype_only", "all"):
            text_proto = self._model_impl().phase5_transform_tensor(text_proto, phase5_state)
            image_proto = self._model_impl().phase5_transform_tensor(image_proto, phase5_state)

        if source == "text":
            return text_proto
        if source == "visual":
            return image_proto

        if isinstance(beta, torch.Tensor):
            beta_value = beta.to(device=text_proto.device, dtype=text_proto.dtype)
            while beta_value.ndim < text_proto.ndim:
                beta_value = beta_value.unsqueeze(-1)
        else:
            beta_value = float(beta)
        return torch.nn.functional.normalize(beta_value * text_proto + (1 - beta_value) * image_proto, dim=-1)


    def _phase7_hubness(self, state_dict, phase5_state, prototypes):
        features = state_dict["images_features"]
        if phase5_state and phase5_state.get("enabled") and phase5_state.get("apply_to") in ("query_only", "all"):
            features = self._model_impl().phase5_transform_tensor(features, phase5_state)
        if features.shape[-1] != prototypes.shape[-1]:
            return torch.zeros(prototypes.shape[0], device=prototypes.device, dtype=prototypes.dtype)
        scores = torch.nn.functional.normalize(features, dim=-1).matmul(
            torch.nn.functional.normalize(prototypes, dim=-1).t()
        )
        winners = scores.argmax(dim=1)
        return torch.bincount(winners, minlength=prototypes.shape[0]).to(device=prototypes.device, dtype=prototypes.dtype)


    def _save_phase1_outputs(self, comparison_rows=None):
        if not self.cfg.TRAINER.BiMC.SAVE_PHASE1_REPORT:
            return {
                "run_dir": None,
                "session_metrics": self.eval_results,
                "beta_session_records": self.beta_session_records,
                "beta_class_records": self.beta_class_records,
                "phase2_records": self.phase2_records,
                "phase5_records": self.phase5_records,
                "phase6_records": self.phase6_records,
                "phase7_records": self.phase7_records,
                "label_prior_records": self.label_prior_records,
            }
        summary = write_phase1_outputs(
            cfg=self.cfg,
            dataset_name=self.data_manager.dataset_name,
            session_metrics=self.eval_results,
            beta_session_records=self.beta_session_records,
            beta_class_records=self.beta_class_records,
            comparison_rows=comparison_rows,
            notes=[
                "Class-wise visual margins use leave-one-out same-class prototypes when at least two support samples exist; singleton classes fall back to the regular visual prototype.",
                "Query-wise beta is computed per test batch from unlabeled query features only.",
            ],
        )
        summary["phase2_records"] = self.phase2_records
        summary["phase5_records"] = self.phase5_records
        summary["phase6_records"] = self.phase6_records
        summary["phase7_records"] = self.phase7_records
        summary["label_prior_records"] = self.label_prior_records
        return summary
    

    def parse_batch(self, batch):
        data = batch['image']
        targets = batch['label']
        data = data.to(self.device)
        targets = targets.to(self.device)
        return data, targets
