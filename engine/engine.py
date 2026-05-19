import torch
import torch.nn as nn
from datasets.data_manager import DatasetManager
from tqdm import tqdm
from utils.evaluator import AccuracyEvaluator
from utils.phase1_fusion import (
    beta_statistics,
    cnn_uses_query_beta,
    compute_class_margin_beta,
    get_active_cnn_mode,
    validate_phase1_options,
)
from utils.phase1_report import write_phase1_outputs
from models.bimc import BiMC
import numpy as np
import time


class Runner:

    def __init__(self, cfg):
        self.cfg = cfg
        validate_phase1_options(cfg)
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

        optional_concat_keys = [
            'cnn_features',
            'cnn_targets',
            'cnn_proto',
            'cnn_projected_proto',
        ]
        for key in optional_concat_keys:
            if all(key in d for d in dict_list):
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
                                                             calibrate_novel_vision_proto=self.cfg.TRAINER.BiMC.VISION_CALIBRATION,)

            state_dict_list.append(current_state_dict)            
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

        opts = self.cfg.TRAINER.BiMC
        beta, beta_values, beta_class_records = self._prepare_session_beta(task_id, state_dict)

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
                                                   cnn_proto=state_dict.get('cnn_proto'),
                                                   return_beta_info=True)
            if beta_info.get("beta") is not None:
                beta_chunks.append(beta_info["beta"])

            all_logits.append(logits)
            all_targets.append(targets)

        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        eval_acc = self.evaluator.calc_accuracy(all_logits, all_targets, task_id) 
        beta_record = {"session": int(task_id)}
        beta_record.update(beta_statistics(beta_chunks))
        self._assert_beta_range(beta_chunks)
        eval_acc["beta_stats"] = beta_record
        self.beta_session_records.append(beta_record)
        self.beta_class_records.extend(beta_class_records)
        if beta_record["beta_count"] > 0:
            print(
                "Beta stats: "
                f"session={task_id}, mode={opts.FUSION_BETA_MODE}, "
                f"cnn_mode={get_active_cnn_mode(opts)}, "
                f"mean={beta_record['beta_mean']:.4f}, "
                f"min={beta_record['beta_min']:.4f}, "
                f"max={beta_record['beta_max']:.4f}"
            )
        print(f"Test acc mean: {eval_acc['mean_acc']}, task-wise acc: {eval_acc['task_acc']}")
        return eval_acc


    def _prepare_session_beta(self, task_id, state_dict):
        opts = self.cfg.TRAINER.BiMC
        num_accumulated_class = max(self.data_manager.class_index_in_task[task_id]) + 1
        if cnn_uses_query_beta(get_active_cnn_mode(opts)):
            return float(self.cfg.DATASET.BETA), None, []

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


    def _save_phase1_outputs(self, comparison_rows=None):
        if not self.cfg.TRAINER.BiMC.SAVE_PHASE1_REPORT:
            return {
                "run_dir": None,
                "session_metrics": self.eval_results,
                "beta_session_records": self.beta_session_records,
                "beta_class_records": self.beta_class_records,
            }
        return write_phase1_outputs(
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
    

    def parse_batch(self, batch):
        data = batch['image']
        targets = batch['label']
        data = data.to(self.device)
        targets = targets.to(self.device)
        return data, targets
