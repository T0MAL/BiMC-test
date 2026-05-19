import argparse
import os
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engine.engine import Runner
from main import setup_cfg
from utils.phase34_report import (
    markdown_table,
    safe_name,
    write_phase34_aggregate,
    write_phase34_run_outputs,
)
from utils.util import set_gpu, set_seed


ABLATIONS = [
    {
        "id": "P34_A0",
        "run_name": "p34_a0_original",
        "settings": {"description": "original"},
        "overrides": {},
    },
    {
        "id": "P34_A1",
        "run_name": "p34_a1_hubness",
        "settings": {"description": "hubness"},
        "overrides": {
            "PHASE3.ENABLED": True,
            "PHASE3.HUBNESS_ENABLED": True,
            "PHASE3.HUBNESS_SOURCE": "mixed",
        },
    },
    {
        "id": "P34_A2",
        "run_name": "p34_a2_temp_scaling",
        "settings": {"description": "temperature scaling"},
        "overrides": {
            "PHASE3.ENABLED": True,
            "PHASE3.TEMP_SCALING_ENABLED": True,
        },
    },
    {
        "id": "P34_A3",
        "run_name": "p34_a3_dynamic_alpha_entropy_margin",
        "settings": {"description": "dynamic alpha entropy_margin"},
        "overrides": {
            "PHASE3.ENABLED": True,
            "PHASE3.DYNAMIC_ALPHA_ENABLED": True,
            "PHASE3.DYNAMIC_ALPHA_MODE": "entropy_margin",
        },
    },
    {
        "id": "P34_A4",
        "run_name": "p34_a4_dynamic_alpha_energy",
        "settings": {"description": "dynamic alpha entropy_margin_energy"},
        "overrides": {
            "PHASE3.ENABLED": True,
            "PHASE3.ENERGY_ENABLED": True,
            "PHASE3.DYNAMIC_ALPHA_ENABLED": True,
            "PHASE3.DYNAMIC_ALPHA_MODE": "entropy_margin_energy",
        },
    },
    {
        "id": "P34_A5",
        "run_name": "p34_a5_hubness_dynamic_alpha",
        "settings": {"description": "hubness + dynamic alpha"},
        "overrides": {
            "PHASE3.ENABLED": True,
            "PHASE3.HUBNESS_ENABLED": True,
            "PHASE3.DYNAMIC_ALPHA_ENABLED": True,
            "PHASE3.DYNAMIC_ALPHA_MODE": "entropy_margin",
        },
    },
    {
        "id": "P34_A6",
        "run_name": "p34_a6_diag_shrinkage_cov",
        "settings": {"description": "diagonal shrinkage covariance"},
        "overrides": {
            "PHASE4.ENABLED": True,
            "PHASE4.COV_MODE": "diag_shrinkage",
            "PHASE4.APPLY_TO_NOVEL": True,
            "PHASE4.REPLACE_NOVEL_NN": False,
            "PHASE4.NOVEL_AUX_COMBINE_MODE": "average",
        },
    },
    {
        "id": "P34_A7",
        "run_name": "p34_a7_base_borrowed_cov",
        "settings": {"description": "base-borrowed covariance"},
        "overrides": {
            "PHASE4.ENABLED": True,
            "PHASE4.COV_MODE": "base_borrowed_diag",
            "PHASE4.APPLY_TO_NOVEL": True,
            "PHASE4.REPLACE_NOVEL_NN": False,
            "PHASE4.NOVEL_AUX_COMBINE_MODE": "average",
        },
    },
    {
        "id": "P34_A8",
        "run_name": "p34_a8_hybrid_cov",
        "settings": {"description": "hybrid covariance"},
        "overrides": {
            "PHASE4.ENABLED": True,
            "PHASE4.COV_MODE": "hybrid_diag",
            "PHASE4.APPLY_TO_NOVEL": True,
            "PHASE4.REPLACE_NOVEL_NN": False,
            "PHASE4.NOVEL_AUX_COMBINE_MODE": "average",
        },
    },
    {
        "id": "P34_A9",
        "run_name": "p34_a9_hubness_dynamic_alpha_hybrid_cov",
        "settings": {"description": "hubness + dynamic alpha + hybrid covariance"},
        "overrides": {
            "PHASE3.ENABLED": True,
            "PHASE3.HUBNESS_ENABLED": True,
            "PHASE3.DYNAMIC_ALPHA_ENABLED": True,
            "PHASE3.DYNAMIC_ALPHA_MODE": "entropy_margin",
            "PHASE4.ENABLED": True,
            "PHASE4.COV_MODE": "hybrid_diag",
            "PHASE4.APPLY_TO_NOVEL": True,
            "PHASE4.NOVEL_AUX_COMBINE_MODE": "average",
        },
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run Phase 3/4 BiMC ablations.")
    parser.add_argument("--dataset", required=True, help="Dataset config path or dataset key.")
    parser.add_argument("--config", default="configs/trainers/bimc.yaml", help="Trainer config path.")
    parser.add_argument("--output-dir", default="results/phase34", help="Directory for Phase 3/4 reports.")
    parser.add_argument("--device", type=str, default=None, help="Override cfg.DEVICE.DEVICE_NAME.")
    parser.add_argument("--gpu-id", type=str, default=None, help="Override cfg.DEVICE.GPU_ID, e.g. 0;")
    parser.add_argument("--seed", type=int, default=None, help="Override cfg.SEED.")
    parser.add_argument("--only", nargs="*", default=None, help="Optional list of ablation IDs, e.g. P34_A1 P34_A8.")
    return parser.parse_args()


def resolve_dataset_cfg(dataset):
    if os.path.isfile(dataset):
        return dataset
    candidate = os.path.join(ROOT, dataset)
    if os.path.isfile(candidate):
        return candidate

    key = dataset.lower().replace("-", "").replace("_", "")
    aliases = {
        "cub": "cub200",
        "cub200": "cub200",
        "cifar": "cifar100",
        "cifar100": "cifar100",
        "mini": "miniimagenet",
        "miniimagenet": "miniimagenet",
    }
    key = aliases.get(key, key)
    candidate = os.path.join(ROOT, "configs", "datasets", f"{key}.yaml")
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError(f"Could not resolve dataset config from {dataset}.")


def resolve_train_cfg(config):
    if os.path.isfile(config):
        return config
    candidate = os.path.join(ROOT, config)
    if os.path.isfile(candidate):
        return candidate
    candidate = os.path.join(ROOT, "configs", "trainers", config)
    if os.path.isfile(candidate):
        return candidate
    if not config.endswith(".yaml"):
        candidate = os.path.join(ROOT, "configs", "trainers", f"{config}.yaml")
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"Could not resolve trainer config from {config}.")


def selected_ablations(only):
    if not only:
        return ABLATIONS
    requested = set()
    for item in only:
        for part in item.split(","):
            if part.strip():
                requested.add(part.strip().upper())
    return [ablation for ablation in ABLATIONS if ablation["id"] in requested or ablation["run_name"].upper() in requested]


def apply_base_cli_cfg(cfg, args):
    cfg = cfg.clone()
    cfg.defrost()
    cfg.OUTPUT_DIR = args.output_dir
    if args.device is not None:
        cfg.DEVICE.DEVICE_NAME = args.device
    if args.gpu_id is not None:
        cfg.DEVICE.GPU_ID = args.gpu_id
    if args.seed is not None:
        cfg.SEED = args.seed
    cfg.freeze()
    return cfg


def apply_ablation_cfg(base_cfg, ablation, output_dir, timestamp):
    cfg = base_cfg.clone()
    cfg.defrost()
    cfg.OUTPUT_DIR = output_dir
    cfg.RUN_NAME = ablation["run_name"]
    cfg.PHASE34_TIMESTAMP = timestamp
    cfg.TRAINER.BiMC.SAVE_PHASE1_REPORT = False
    _reset_phase34_flags(cfg)
    for dotted_key, value in ablation["overrides"].items():
        _set_bimc_nested(cfg, dotted_key, value)
    cfg.freeze()
    return cfg


def build_cached_task_states(runner):
    state_dict_list = []
    merged_states = []
    print(f"Building task statistics once for {runner.data_manager.num_tasks} tasks.")
    for task_id in range(runner.data_manager.num_tasks):
        runner.model.eval()
        current_class_name = np.array(runner.data_manager.class_names)[runner.data_manager.class_index_in_task[task_id]]
        loader = runner.data_manager.get_dataloader(task_id, source="train", mode="test", accumulate_past=False)
        current_state_dict = runner._model_impl().build_task_statistics(
            current_class_name,
            loader,
            class_index=runner.data_manager.class_index_in_task[task_id],
            calibrate_novel_vision_proto=runner.cfg.TRAINER.BiMC.VISION_CALIBRATION,
        )
        state_dict_list.append(current_state_dict)
        merged_states.append(runner.merge_dicts(state_dict_list))
    return merged_states


def run_ablation_from_cache(runner, cfg, merged_states):
    _reset_runner_records(runner, cfg)
    start = time.time()
    print(f"Running {cfg.RUN_NAME}")
    for task_id, state_dict in enumerate(merged_states):
        runner.model.eval()
        acc = runner.inference_task_covariance(task_id, state_dict)
        print(f"=> Task [{task_id}], Acc: {acc['mean_acc']:.3f}")
        runner.acc_list.append(round(acc["mean_acc"], 3))
        runner.task_acc_list.append(acc["task_acc"])
        runner.eval_results.append(acc)
    summary = runner._save_phase1_outputs()
    summary["status"] = "ok"
    summary["elapsed_seconds"] = round(time.time() - start, 3)
    return summary


def failed_summary(error):
    return {
        "status": "failed",
        "error": error,
        "session_metrics": [],
        "beta_session_records": [],
        "beta_class_records": [],
        "phase3_score_records": [],
        "phase4_cov_records": [],
        "hubness_class_records": [],
        "dynamic_alpha_records": [],
        "covariance_class_records": [],
    }


def main():
    args = parse_args()
    data_cfg = resolve_dataset_cfg(args.dataset)
    train_cfg = resolve_train_cfg(args.config)
    base_cfg = apply_base_cli_cfg(setup_cfg(data_cfg, train_cfg), args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = os.path.abspath(args.output_dir)
    parent_dir = os.path.join(output_root, safe_name(base_cfg.DATASET.NAME), timestamp)

    set_seed(base_cfg.SEED)
    if base_cfg.DEVICE.DEVICE_NAME.startswith("cuda"):
        set_gpu(base_cfg.DEVICE.GPU_ID)

    runner = Runner(base_cfg)
    merged_states = build_cached_task_states(runner)

    run_records = []
    for ablation in selected_ablations(args.only):
        cfg = apply_ablation_cfg(base_cfg, ablation, output_root, timestamp)
        set_seed(cfg.SEED)
        try:
            summary = run_ablation_from_cache(runner, cfg, merged_states)
        except Exception:
            error = traceback.format_exc()
            print(f"{ablation['run_name']} failed:\n{error}")
            summary = failed_summary(error)
        run_dir = os.path.join(parent_dir, ablation["run_name"])
        run_record = write_phase34_run_outputs(run_dir, base_cfg.DATASET.NAME, cfg, summary, ablation)
        run_records.append(run_record)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    aggregate = write_phase34_aggregate(parent_dir, base_cfg.DATASET.NAME, base_cfg, run_records)
    print(markdown_table(aggregate["comparison_rows"]))
    print(f"Phase 3/4 aggregate report: {aggregate['report_path']}")


def _reset_phase34_flags(cfg):
    p3 = cfg.TRAINER.BiMC.PHASE3
    p3.ENABLED = False
    p3.HUBNESS_ENABLED = False
    p3.TEMP_SCALING_ENABLED = False
    p3.ENERGY_ENABLED = False
    p3.DYNAMIC_ALPHA_ENABLED = False

    p4 = cfg.TRAINER.BiMC.PHASE4
    p4.ENABLED = False
    p4.COV_MODE = "original"
    p4.APPLY_TO_NOVEL = True
    p4.APPLY_TO_BASE = False
    p4.REPLACE_NOVEL_NN = False
    p4.NOVEL_AUX_COMBINE_MODE = "alpha"


def _set_bimc_nested(cfg, dotted_key, value):
    node = cfg.TRAINER.BiMC
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        node = getattr(node, part)
    setattr(node, parts[-1], value)


def _reset_runner_records(runner, cfg):
    runner.cfg = cfg
    model = runner._model_impl()
    model.cfg = cfg
    runner.acc_list = []
    runner.task_acc_list = []
    runner.eval_results = []
    runner.beta_session_records = []
    runner.beta_class_records = []
    runner.phase3_score_records = []
    runner.phase4_cov_records = []
    runner.hubness_class_records = []
    runner.dynamic_alpha_records = []
    runner.covariance_class_records = []


if __name__ == "__main__":
    main()
