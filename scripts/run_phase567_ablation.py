import argparse
import os
import sys
import traceback
from datetime import datetime

import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engine.engine import Runner
from main import setup_cfg
from utils.phase567_report import (
    failed_phase567_summary,
    write_phase567_ablation_summary,
    write_phase567_outputs,
)
from utils.util import set_gpu, set_seed


ABLATIONS = [
    {
        "ablation_id": "P567_A0",
        "run_name": "p567_a0_original",
        "phase5": {"ENABLED": False},
        "phase6": {"ENABLED": False},
        "phase7": {"ENABLED": False},
    },
    {
        "ablation_id": "P567_A1",
        "run_name": "p567_a1_common_direction",
        "phase5": {"ENABLED": True, "SPACE_TRANSFORM": "common_direction_removal", "APPLY_TO": "all"},
        "phase6": {"ENABLED": False},
        "phase7": {"ENABLED": False},
    },
    {
        "ablation_id": "P567_A2",
        "run_name": "p567_a2_whitening_diag",
        "phase5": {"ENABLED": True, "SPACE_TRANSFORM": "whitening", "WHITENING_DIAG_ONLY": True, "APPLY_TO": "all"},
        "phase6": {"ENABLED": False},
        "phase7": {"ENABLED": False},
    },
    {
        "ablation_id": "P567_A3",
        "run_name": "p567_a3_lda_shrinkage",
        "phase5": {"ENABLED": True, "SPACE_TRANSFORM": "lda_shrinkage", "LDA_DIM": 256},
        "phase6": {"ENABLED": False},
        "phase7": {"ENABLED": False},
    },
    {
        "ablation_id": "P567_A4",
        "run_name": "p567_a4_ot_text_image",
        "phase5": {"ENABLED": False},
        "phase6": {"ENABLED": True, "ALIGNMENT_MODE": "ot_text_image"},
        "phase7": {"ENABLED": False},
    },
    {
        "ablation_id": "P567_A5",
        "run_name": "p567_a5_support_aware_text",
        "phase5": {"ENABLED": False},
        "phase6": {"ENABLED": True, "ALIGNMENT_MODE": "support_aware_text"},
        "phase7": {"ENABLED": False},
    },
    {
        "ablation_id": "P567_A6",
        "run_name": "p567_a6_label_prior",
        "phase5": {"ENABLED": False},
        "phase6": {"ENABLED": True, "ALIGNMENT_MODE": "label_prior_correction", "LABEL_PRIOR_ENABLED": True},
        "phase7": {"ENABLED": False},
    },
    {
        "ablation_id": "P567_A7",
        "run_name": "p567_a7_prototype_repulsion",
        "phase5": {"ENABLED": False},
        "phase6": {"ENABLED": False},
        "phase7": {"ENABLED": True, "SEPARATION_MODE": "prototype_repulsion"},
    },
    {
        "ablation_id": "P567_A8",
        "run_name": "p567_a8_graph_highpass",
        "phase5": {"ENABLED": False},
        "phase6": {"ENABLED": False},
        "phase7": {"ENABLED": True, "SEPARATION_MODE": "graph_highpass"},
    },
    {
        "ablation_id": "P567_A9",
        "run_name": "p567_a9_cdr_support_text",
        "phase5": {"ENABLED": True, "SPACE_TRANSFORM": "common_direction_removal"},
        "phase6": {"ENABLED": True, "ALIGNMENT_MODE": "support_aware_text"},
        "phase7": {"ENABLED": False},
    },
    {
        "ablation_id": "P567_A10",
        "run_name": "p567_a10_whitening_graph",
        "phase5": {"ENABLED": True, "SPACE_TRANSFORM": "whitening", "WHITENING_DIAG_ONLY": True},
        "phase6": {"ENABLED": False},
        "phase7": {"ENABLED": True, "SEPARATION_MODE": "graph_highpass"},
    },
    {
        "ablation_id": "P567_A11",
        "run_name": "p567_a11_ot_label_prior",
        "phase5": {"ENABLED": False},
        "phase6": {"ENABLED": True, "ALIGNMENT_MODE": "ot_text_image", "LABEL_PRIOR_ENABLED": True},
        "phase7": {"ENABLED": False},
    },
    {
        "ablation_id": "P567_A12",
        "run_name": "p567_a12_best_combo_safe",
        "phase5": {"ENABLED": True, "SPACE_TRANSFORM": "common_direction_removal"},
        "phase6": {"ENABLED": True, "ALIGNMENT_MODE": "support_aware_text"},
        "phase7": {"ENABLED": True, "SEPARATION_MODE": "graph_highpass"},
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run Phase 5/6/7 BiMC ablations.")
    parser.add_argument("--dataset", required=True, help="Dataset config path or dataset key.")
    parser.add_argument("--config", default="configs/trainers/bimc.yaml", help="Trainer config path.")
    parser.add_argument("--output-dir", default="results/phase567", help="Directory for Phase 5/6/7 reports.")
    parser.add_argument("--device", type=str, default=None, help="Override cfg.DEVICE.DEVICE_NAME, e.g. cuda or cpu.")
    parser.add_argument("--gpu-id", type=str, default=None, help="Override cfg.DEVICE.GPU_ID, e.g. 0;")
    parser.add_argument("--seed", type=int, default=None, help="Override cfg.SEED for all ablations.")
    parser.add_argument("--only", nargs="*", default=None, help="Optional ablation IDs/names, e.g. P567_A1 P567_A8.")
    parser.add_argument("--continue-on-error", dest="continue_on_error", action="store_true", default=True)
    parser.add_argument("--no-continue-on-error", dest="continue_on_error", action="store_false")
    parser.add_argument("--timestamp", type=str, default=None, help=argparse.SUPPRESS)
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
        ablation for ablation in ABLATIONS
        if _normalize_ablation_key(ablation["ablation_id"]) in requested
        or _normalize_ablation_key(ablation["run_name"]) in requested
    ]
    if not selected:
        raise ValueError(f"No Phase 5/6/7 ablations matched --only={only}.")
    baseline = ABLATIONS[0]
    if baseline not in selected:
        selected.insert(0, baseline)
    return selected


def apply_ablation_cfg(base_cfg, ablation, args, timestamp):
    cfg = base_cfg.clone()
    cfg.defrost()
    cfg.OUTPUT_DIR = args.output_dir
    cfg.PHASE567_TIMESTAMP = timestamp
    cfg.RUN_NAME = ablation["run_name"]

    cfg.TRAINER.BiMC.FUSION_BETA_MODE = "fixed"
    cfg.TRAINER.BiMC.FUSION_GEOMETRY = "linear"
    cfg.TRAINER.BiMC.SAVE_PHASE1_REPORT = False
    cfg.TRAINER.BiMC.PHASE2.ENABLED = False
    cfg.TRAINER.BiMC.PHASE2.SAVE_PHASE2_REPORT = False

    _apply_nested_options(cfg.TRAINER.BiMC.PHASE5, {"ENABLED": False, "SPACE_TRANSFORM": "none"})
    _apply_nested_options(cfg.TRAINER.BiMC.PHASE6, {"ENABLED": False, "ALIGNMENT_MODE": "none", "LABEL_PRIOR_ENABLED": False})
    _apply_nested_options(cfg.TRAINER.BiMC.PHASE7, {"ENABLED": False, "SEPARATION_MODE": "none"})
    _apply_nested_options(cfg.TRAINER.BiMC.PHASE5, ablation["phase5"])
    _apply_nested_options(cfg.TRAINER.BiMC.PHASE6, ablation["phase6"])
    _apply_nested_options(cfg.TRAINER.BiMC.PHASE7, ablation["phase7"])

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
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    ablations = select_ablations(args.only)

    run_records = []
    for ablation in ablations:
        cfg = apply_ablation_cfg(base_cfg, ablation, args, timestamp)
        print(f"Running {ablation['run_name']}")
        try:
            set_seed(cfg.SEED)
            if cfg.DEVICE.DEVICE_NAME.startswith("cuda"):
                set_gpu(cfg.DEVICE.GPU_ID)
            runner = Runner(cfg)
            runner_summary = runner.run()
            phase_summary = write_phase567_outputs(
                cfg=cfg,
                dataset_name=runner.data_manager.dataset_name,
                session_metrics=runner_summary["session_metrics"],
                phase5_records=runner_summary["phase5_records"],
                phase6_records=runner_summary["phase6_records"],
                phase7_records=runner_summary["phase7_records"],
                label_prior_records=runner_summary["label_prior_records"],
                status="success",
            )
            run_records.append({"cfg": cfg, "summary": phase_summary})
            del runner
        except Exception as exc:
            traceback.print_exc()
            failed = failed_phase567_summary(cfg, base_cfg.DATASET.NAME, ablation["run_name"], error=exc)
            run_records.append({"cfg": cfg, "summary": failed})
            if not args.continue_on_error:
                raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summaries = [record["summary"] for record in run_records]
    parent_dir = os.path.dirname(summaries[0]["run_dir"]) if summaries else os.path.join(args.output_dir, timestamp)
    aggregate = write_phase567_ablation_summary(
        parent_dir,
        base_cfg.DATASET.NAME,
        base_cfg,
        summaries,
        manifest={
            "command": " ".join(sys.argv),
            "dataset_cfg": data_cfg,
            "trainer_cfg": train_cfg,
            "continue_on_error": bool(args.continue_on_error),
        },
    )

    rows = aggregate["comparison_rows"]
    for record, row in zip(run_records, rows):
        if record["summary"].get("status") != "success":
            continue
        comparison_rows = [rows[0]] if row["is_baseline"] else [rows[0], row]
        write_phase567_outputs(
            cfg=record["cfg"],
            dataset_name=base_cfg.DATASET.NAME,
            session_metrics=record["summary"]["session_metrics"],
            phase5_records=record["summary"]["phase5_records"],
            phase6_records=record["summary"]["phase6_records"],
            phase7_records=record["summary"]["phase7_records"],
            label_prior_records=record["summary"]["label_prior_records"],
            comparison_rows=comparison_rows,
            notes=[
                "Phase 5/6/7 ablations force fixed linear BiMC fusion.",
                "Covariance and KNN support branches remain in the original feature space; Phase 5/7 affect fused-prototype cosine scoring.",
                "Label-prior correction uses unlabeled session predictions and is marked transductive.",
            ],
        )

    print_markdown_table(rows)
    print(f"Phase 5/6/7 report: {aggregate['report_path']}")


def print_markdown_table(rows):
    print("")
    print("| Run | Status | Avg | Final | Delta Avg | Delta Final | Transductive | Suspicious |")
    print("| --- | --- | ---: | ---: | ---: | ---: | --- | --- |")
    for row in rows:
        print(
            "| "
            f"{row['run_name']} | {row['status']} | {_fmt(row['average_accuracy'])} | "
            f"{_fmt(row['final_accuracy'])} | {_fmt(row['delta_avg'])} | {_fmt(row['delta_final'])} | "
            f"{row['transductive']} | {row['suspicious_drop']} |"
        )


def _apply_nested_options(node, values):
    for key, value in values.items():
        setattr(node, key, value)


def _normalize_ablation_key(value):
    value = value.lower().replace("-", "_")
    if value.startswith("p567_"):
        return value
    if value.startswith("a"):
        return f"p567_{value}"
    if value.isdigit():
        return f"p567_a{value}"
    return value


def _fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    return f"{float(value):.3f}"


if __name__ == "__main__":
    main()
