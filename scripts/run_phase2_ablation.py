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
from utils.phase2_report import write_phase2_ablation_summary, write_phase2_outputs
from utils.util import set_gpu, set_seed


ABLATIONS = [
    {
        "ablation_id": "P2_A0",
        "run_name": "p2_a0_original",
        "phase2_enabled": False,
        "visual_proto_mode": "mean",
        "text_proto_mode": "mean",
        "dynamic_lambda_i": False,
    },
    {
        "ablation_id": "P2_A1",
        "run_name": "p2_a1_robust_visual",
        "phase2_enabled": True,
        "visual_proto_mode": "robust_weighted",
        "text_proto_mode": "mean",
        "dynamic_lambda_i": False,
    },
    {
        "ablation_id": "P2_A2",
        "run_name": "p2_a2_shrinkage_visual",
        "phase2_enabled": True,
        "visual_proto_mode": "shrinkage_base_prior",
        "text_proto_mode": "mean",
        "dynamic_lambda_i": False,
    },
    {
        "ablation_id": "P2_A3",
        "run_name": "p2_a3_dynamic_lambda",
        "phase2_enabled": True,
        "visual_proto_mode": "mean",
        "text_proto_mode": "mean",
        "dynamic_lambda_i": True,
    },
    {
        "ablation_id": "P2_A4",
        "run_name": "p2_a4_desc_discriminative",
        "phase2_enabled": True,
        "visual_proto_mode": "mean",
        "text_proto_mode": "discriminative_reweight",
        "dynamic_lambda_i": False,
    },
    {
        "ablation_id": "P2_A5",
        "run_name": "p2_a5_desc_visual_grounded",
        "phase2_enabled": True,
        "visual_proto_mode": "mean",
        "text_proto_mode": "visual_grounded_reweight",
        "dynamic_lambda_i": False,
    },
    {
        "ablation_id": "P2_A6",
        "run_name": "p2_a6_desc_combined",
        "phase2_enabled": True,
        "visual_proto_mode": "mean",
        "text_proto_mode": "combined_reweight",
        "dynamic_lambda_i": False,
    },
    {
        "ablation_id": "P2_A7",
        "run_name": "p2_a7_shrinkage_desc_combined",
        "phase2_enabled": True,
        "visual_proto_mode": "shrinkage_base_prior",
        "text_proto_mode": "combined_reweight",
        "dynamic_lambda_i": False,
    },
    {
        "ablation_id": "P2_A8",
        "run_name": "p2_a8_shrinkage_dynamic_lambda_desc_combined",
        "phase2_enabled": True,
        "visual_proto_mode": "shrinkage_base_prior",
        "text_proto_mode": "combined_reweight",
        "dynamic_lambda_i": True,
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run Phase Two BiMC prototype ablations.")
    parser.add_argument("--dataset", required=True, help="Dataset config path or dataset key.")
    parser.add_argument("--config", default="configs/trainers/bimc.yaml", help="Trainer config path.")
    parser.add_argument("--output-dir", default="results/phase2", help="Directory for phase-two reports.")
    parser.add_argument("--device", type=str, default=None, help="Override cfg.DEVICE.DEVICE_NAME, e.g. cuda or cpu.")
    parser.add_argument("--gpu-id", type=str, default=None, help="Override cfg.DEVICE.GPU_ID, e.g. 0;")
    parser.add_argument("--seed", type=int, default=None, help="Override cfg.SEED for all ablations.")
    parser.add_argument("--only", nargs="*", default=None, help="Optional ablation IDs/names, e.g. P2_A1 P2_A4.")
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


def select_ablations(only):
    if not only:
        return ABLATIONS

    requested = set()
    for item in only:
        for piece in item.split(","):
            piece = piece.strip()
            if piece:
                requested.add(_normalize_ablation_key(piece))

    selected = [
        ablation
        for ablation in ABLATIONS
        if _normalize_ablation_key(ablation["ablation_id"]) in requested
        or _normalize_ablation_key(ablation["run_name"]) in requested
    ]
    if not selected:
        raise ValueError(f"No Phase Two ablations matched --only={only}.")

    baseline = ABLATIONS[0]
    if baseline not in selected:
        selected.insert(0, baseline)
    return selected


def apply_ablation_cfg(base_cfg, ablation, args, timestamp):
    cfg = base_cfg.clone()
    cfg.defrost()
    cfg.OUTPUT_DIR = args.output_dir
    cfg.PHASE2_TIMESTAMP = timestamp
    cfg.RUN_NAME = ablation["run_name"]

    # Phase Two is intentionally independent of Phase One dynamic beta.
    cfg.TRAINER.BiMC.FUSION_BETA_MODE = "fixed"
    cfg.TRAINER.BiMC.FUSION_GEOMETRY = "linear"

    cfg.TRAINER.BiMC.SAVE_PHASE1_REPORT = False
    cfg.TRAINER.BiMC.PHASE2.SAVE_PHASE2_REPORT = True
    cfg.TRAINER.BiMC.PHASE2.ENABLED = ablation["phase2_enabled"]
    cfg.TRAINER.BiMC.PHASE2.VISUAL_PROTO_MODE = ablation["visual_proto_mode"]
    cfg.TRAINER.BiMC.PHASE2.TEXT_PROTO_MODE = ablation["text_proto_mode"]
    cfg.TRAINER.BiMC.PHASE2.DYNAMIC_LAMBDA_I = ablation["dynamic_lambda_i"]

    if args.seed is not None:
        cfg.SEED = args.seed
    if args.device is not None:
        cfg.DEVICE.DEVICE_NAME = args.device
    if args.gpu_id is not None:
        cfg.DEVICE.GPU_ID = args.gpu_id
    cfg.freeze()
    return cfg


def main():
    args = parse_args()
    data_cfg = resolve_dataset_cfg(args.dataset)
    train_cfg = resolve_train_cfg(args.config)
    base_cfg = setup_cfg(data_cfg, train_cfg)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ablations = select_ablations(args.only)

    run_records = []
    for ablation in ablations:
        cfg = apply_ablation_cfg(base_cfg, ablation, args, timestamp)
        set_seed(cfg.SEED)
        if cfg.DEVICE.DEVICE_NAME.startswith("cuda"):
            set_gpu(cfg.DEVICE.GPU_ID)

        print(
            "Running "
            f"{ablation['run_name']}: visual={ablation['visual_proto_mode']}, "
            f"text={ablation['text_proto_mode']}, dynamic_lambda={ablation['dynamic_lambda_i']}"
        )
        runner = Runner(cfg)
        runner_summary = runner.run()
        phase2_summary = write_phase2_outputs(
            cfg=cfg,
            dataset_name=runner.data_manager.dataset_name,
            session_metrics=runner_summary["session_metrics"],
            phase2_records=runner_summary["phase2_records"],
        )
        run_records.append({"cfg": cfg, "summary": phase2_summary})
        del runner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summaries = [record["summary"] for record in run_records]
    parent_dir = os.path.dirname(summaries[0]["run_dir"])
    aggregate = write_phase2_ablation_summary(parent_dir, base_cfg.DATASET.NAME, base_cfg, summaries)

    rows = aggregate["comparison_rows"]
    for record, row in zip(run_records, rows):
        comparison_rows = [rows[0]] if row["is_baseline"] else [rows[0], row]
        write_phase2_outputs(
            cfg=record["cfg"],
            dataset_name=base_cfg.DATASET.NAME,
            session_metrics=record["summary"]["session_metrics"],
            phase2_records=record["summary"]["phase2_records"],
            comparison_rows=comparison_rows,
            notes=[
                "Phase Two ablations keep fusion_beta_mode=fixed and fusion_geometry=linear.",
                "Prototype improvements use support-set features, class-name text features, and stored base prototypes only.",
            ],
        )

    print_markdown_table(rows)
    print(f"Phase Two report: {aggregate['report_path']}")


def print_markdown_table(rows):
    print("")
    print("| Run | Avg | Final | PD | Delta Avg | Delta Final | Delta PD |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        print(
            "| "
            f"{row['run_name']} | {_fmt(row['average_accuracy'])} | {_fmt(row['final_accuracy'])} | "
            f"{_fmt(row['performance_degradation'])} | {_fmt(row['delta_avg'])} | "
            f"{_fmt(row['delta_final'])} | {_fmt(row['delta_pd'])} |"
        )


def _normalize_ablation_key(value):
    value = value.lower().replace("-", "_")
    if value.startswith("p2_"):
        return value
    if value.startswith("a"):
        return f"p2_{value}"
    if value.isdigit():
        return f"p2_a{value}"
    return value


def _fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    return f"{float(value):.3f}"


if __name__ == "__main__":
    main()
