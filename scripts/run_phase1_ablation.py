import argparse
import os
import sys
from datetime import datetime

import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engine.engine import Runner
from main import setup_cfg
from utils.phase1_report import write_phase1_ablation_summary, write_phase1_outputs
from utils.util import set_gpu, set_seed


ABLATIONS = [
    {
        "run_name": "A0_original",
        "fusion_beta_mode": "fixed",
        "fusion_geometry": "linear",
        "reliability_mode": "entropy_margin",
    },
    {
        "run_name": "A1_fixed_slerp",
        "fusion_beta_mode": "fixed",
        "fusion_geometry": "slerp",
        "reliability_mode": "entropy_margin",
    },
    {
        "run_name": "A2_class_margin_linear",
        "fusion_beta_mode": "class_margin",
        "fusion_geometry": "linear",
        "reliability_mode": "entropy_margin",
    },
    {
        "run_name": "A3_class_margin_slerp",
        "fusion_beta_mode": "class_margin",
        "fusion_geometry": "slerp",
        "reliability_mode": "entropy_margin",
    },
    {
        "run_name": "A4_query_reliability_linear",
        "fusion_beta_mode": "query_reliability",
        "fusion_geometry": "linear",
        "reliability_mode": "entropy_margin",
    },
    {
        "run_name": "A5_query_reliability_slerp",
        "fusion_beta_mode": "query_reliability",
        "fusion_geometry": "slerp",
        "reliability_mode": "entropy_margin",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run Phase One BiMC ablations.")
    parser.add_argument("--dataset", required=True, help="Dataset config path or dataset key.")
    parser.add_argument("--config", default="configs/trainers/bimc.yaml", help="Trainer config path.")
    parser.add_argument("--output-dir", default="results/phase1", help="Directory for phase-one reports.")
    parser.add_argument("--sessions", type=str, default=None, help="Accepted for compatibility; current pipeline runs all sessions.")
    parser.add_argument("--device", type=str, default=None, help="Override cfg.DEVICE.DEVICE_NAME, e.g. cuda or cuda:0.")
    parser.add_argument("--gpu-id", type=str, default=None, help="Override cfg.DEVICE.GPU_ID, e.g. 0;")
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


def apply_ablation_cfg(base_cfg, ablation, args, timestamp):
    cfg = base_cfg.clone()
    cfg.defrost()
    cfg.OUTPUT_DIR = args.output_dir
    cfg.PHASE1_TIMESTAMP = timestamp
    cfg.RUN_NAME = ablation["run_name"]
    cfg.TRAINER.BiMC.FUSION_BETA_MODE = ablation["fusion_beta_mode"]
    cfg.TRAINER.BiMC.FUSION_GEOMETRY = ablation["fusion_geometry"]
    cfg.TRAINER.BiMC.RELIABILITY_MODE = ablation["reliability_mode"]
    cfg.TRAINER.BiMC.SAVE_PHASE1_REPORT = True
    if args.device is not None:
        cfg.DEVICE.DEVICE_NAME = args.device
    if args.gpu_id is not None:
        cfg.DEVICE.GPU_ID = args.gpu_id
    cfg.freeze()
    return cfg


def main():
    args = parse_args()
    if args.sessions is not None:
        print("--sessions was provided, but this repository's runner does not expose session filtering; running all sessions.")

    data_cfg = resolve_dataset_cfg(args.dataset)
    train_cfg = resolve_train_cfg(args.config)
    base_cfg = setup_cfg(data_cfg, train_cfg)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_records = []
    for ablation in ABLATIONS:
        cfg = apply_ablation_cfg(base_cfg, ablation, args, timestamp)
        set_seed(cfg.SEED)
        if cfg.DEVICE.DEVICE_NAME.startswith("cuda"):
            set_gpu(cfg.DEVICE.GPU_ID)

        print(f"Running {ablation['run_name']}: {ablation['fusion_beta_mode']} + {ablation['fusion_geometry']}")
        runner = Runner(cfg)
        summary = runner.run()
        run_records.append({"cfg": cfg, "summary": summary})
        del runner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summaries = [record["summary"] for record in run_records]
    parent_dir = os.path.dirname(summaries[0]["run_dir"])
    aggregate = write_phase1_ablation_summary(parent_dir, base_cfg.DATASET.NAME, base_cfg, summaries)

    rows = aggregate["comparison_rows"]
    for record, row in zip(run_records, rows):
        comparison_rows = [rows[0]] if row["is_baseline"] else [rows[0], row]
        write_phase1_outputs(
            cfg=record["cfg"],
            dataset_name=base_cfg.DATASET.NAME,
            session_metrics=record["summary"]["session_metrics"],
            beta_session_records=record["summary"]["beta_session_records"],
            beta_class_records=record["summary"]["beta_class_records"],
            comparison_rows=comparison_rows,
            notes=[
                "Class-wise visual margins use leave-one-out same-class prototypes when at least two support samples exist; singleton classes fall back to the regular visual prototype.",
                "Query-wise beta is computed per test batch from unlabeled query features only.",
            ],
        )

    print(f"Phase One report: {aggregate['report_path']}")


if __name__ == "__main__":
    main()
