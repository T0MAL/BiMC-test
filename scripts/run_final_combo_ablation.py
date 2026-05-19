import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from time import perf_counter


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.ablation_runner_utils import (
    deep_update,
    load_config,
    normalize_metric_row,
    resolve_dataset_cfg,
    resolve_train_cfg,
    run_eval_with_config,
    save_config_snapshot,
    to_plain,
)


CSV_COLUMNS = [
    "dataset",
    "combo_id",
    "combo_name",
    "combo_set",
    "transductive",
    "risky",
    "average_accuracy",
    "final_accuracy",
    "performance_degradation",
    "base_final_accuracy",
    "novel_final_accuracy",
    "delta_avg_vs_original",
    "delta_final_vs_original",
    "delta_pd_vs_original",
    "status",
    "duration_sec",
    "output_dir",
    "report_path",
    "error_message",
]


PHASES = {
    2: {
        "label": "Phase 2 prototype improvements",
        "modules": ("utils.phase2_prototypes",),
    },
    3: {
        "label": "Phase 3 score reliability",
        "modules": ("utils.phase3_scores",),
    },
    4: {
        "label": "Phase 4 covariance",
        "modules": ("utils.phase4_covariance",),
    },
    5: {
        "label": "Phase 5 space transform",
        "modules": ("utils.phase5_space",),
    },
    6: {
        "label": "Phase 6 alignment",
        "modules": ("utils.phase6_alignment",),
    },
    7: {
        "label": "Phase 7 prototype separation",
        "modules": ("utils.phase7_separation",),
    },
}


@dataclass(frozen=True)
class Combo:
    combo_id: str
    name: str
    purpose: str
    overrides: dict
    required_phases: tuple = ()
    disable_phases: tuple = ()
    optional_overrides: tuple = ()
    transductive: bool = False
    risky: bool = False
    components: tuple = ()


def _base_phase1(beta_mode="fixed", geometry="linear", reliability_mode="entropy_margin"):
    return {
        "PHASE1.FUSION_BETA_MODE": beta_mode,
        "PHASE1.FUSION_GEOMETRY": geometry,
        "PHASE1.RELIABILITY_MODE": reliability_mode,
    }


FINAL_COMBOS = (
    Combo(
        combo_id="C0_ORIGINAL",
        name="Original BiMC",
        purpose="Original BiMC with fixed beta and linear fusion.",
        overrides=_base_phase1("fixed", "linear"),
        disable_phases=(2, 3, 4, 5, 6, 7),
        components=("original",),
    ),
    Combo(
        combo_id="C1_SAFE_PROTOTYPE",
        name="Safe prototype",
        purpose="Test whether prototype quality alone improves BiMC.",
        overrides={
            **_base_phase1("fixed", "linear"),
            "PHASE2.ENABLED": True,
            "PHASE2.VISUAL_PROTO_MODE": "shrinkage_base_prior",
            "PHASE2.TEXT_PROTO_MODE": "combined_reweight",
            "PHASE2.DYNAMIC_LAMBDA_I": False,
        },
        required_phases=(2,),
        disable_phases=(3, 4, 5, 6, 7),
        components=("prototype",),
    ),
    Combo(
        combo_id="C2_SAFE_PROTOTYPE_SLERP",
        name="Safe prototype SLERP",
        purpose="Test prototype quality with spherical fusion only.",
        overrides={
            **_base_phase1("fixed", "slerp"),
            "PHASE2.ENABLED": True,
            "PHASE2.VISUAL_PROTO_MODE": "shrinkage_base_prior",
            "PHASE2.TEXT_PROTO_MODE": "combined_reweight",
        },
        required_phases=(2,),
        disable_phases=(3, 4, 5, 6, 7),
        components=("prototype", "slerp"),
    ),
    Combo(
        combo_id="C3_SAFE_PROTOTYPE_SPACE",
        name="Safe prototype space",
        purpose="Test prototype improvement plus space de-crowding.",
        overrides={
            **_base_phase1("fixed", "linear"),
            "PHASE2.ENABLED": True,
            "PHASE2.VISUAL_PROTO_MODE": "shrinkage_base_prior",
            "PHASE2.TEXT_PROTO_MODE": "combined_reweight",
            "PHASE5.ENABLED": True,
            "PHASE5.SPACE_TRANSFORM": "common_direction_removal",
            "PHASE5.APPLY_TO": "all",
        },
        required_phases=(2, 5),
        disable_phases=(4, 6, 7),
        optional_overrides=(
            ("PHASE3.ENABLED", True),
            ("PHASE3.HUBNESS_ENABLED", True),
        ),
        components=("prototype", "space", "hubness"),
    ),
    Combo(
        combo_id="C4_UNCERTAINTY_AWARE",
        name="Uncertainty aware",
        purpose="Test score reliability and dynamic ensemble weighting.",
        overrides={
            **_base_phase1("query_reliability", "slerp", "entropy_margin"),
            "PHASE3.ENABLED": True,
            "PHASE3.TEMP_SCALING_ENABLED": True,
            "PHASE3.ENERGY_ENABLED": True,
            "PHASE3.DYNAMIC_ALPHA_ENABLED": True,
            "PHASE3.DYNAMIC_ALPHA_MODE": "entropy_margin_energy",
        },
        required_phases=(3,),
        disable_phases=(2, 4, 5, 6, 7),
        risky=True,
        components=("uncertainty", "dynamic_beta"),
    ),
    Combo(
        combo_id="C5_PROTOTYPE_SPACE_SYSTEM",
        name="Prototype space system",
        purpose="Test whether prototype-space correction helps.",
        overrides={
            **_base_phase1("fixed", "slerp"),
            "PHASE2.ENABLED": True,
            "PHASE2.VISUAL_PROTO_MODE": "shrinkage_base_prior",
            "PHASE2.TEXT_PROTO_MODE": "combined_reweight",
            "PHASE5.ENABLED": True,
            "PHASE5.SPACE_TRANSFORM": "common_direction_removal",
            "PHASE5.APPLY_TO": "all",
            "PHASE3.ENABLED": True,
            "PHASE3.HUBNESS_ENABLED": True,
            "PHASE3.HUBNESS_SOURCE": "mixed",
        },
        required_phases=(2, 3, 5),
        disable_phases=(4, 6, 7),
        components=("prototype", "space", "hubness"),
    ),
    Combo(
        combo_id="C6_NOVEL_CLASS_SYSTEM",
        name="Novel class system",
        purpose="Improve novel-class performance.",
        overrides={
            **_base_phase1("fixed", "linear"),
            "PHASE2.ENABLED": True,
            "PHASE2.VISUAL_PROTO_MODE": "shrinkage_base_prior",
            "PHASE2.DYNAMIC_LAMBDA_I": True,
            "PHASE4.ENABLED": True,
            "PHASE4.COV_MODE": "hybrid_diag",
            "PHASE4.APPLY_TO_NOVEL": True,
            "PHASE4.REPLACE_NOVEL_NN": False,
            "PHASE4.NOVEL_AUX_COMBINE_MODE": "average",
            "PHASE3.ENABLED": True,
            "PHASE3.DYNAMIC_ALPHA_ENABLED": True,
            "PHASE3.DYNAMIC_ALPHA_MODE": "entropy_margin",
        },
        required_phases=(2, 3, 4),
        disable_phases=(5, 6, 7),
        components=("prototype", "uncertainty", "covariance"),
    ),
    Combo(
        combo_id="C7_CUB_FINEGRAINED_SAFE",
        name="CUB fine-grained safe",
        purpose="Fine-grained text/vision alignment for CUB-like datasets.",
        overrides={
            **_base_phase1("fixed", "slerp"),
            "PHASE2.ENABLED": True,
            "PHASE2.TEXT_PROTO_MODE": "combined_reweight",
            "PHASE2.VISUAL_PROTO_MODE": "shrinkage_base_prior",
            "PHASE6.ENABLED": True,
            "PHASE6.ALIGNMENT_MODE": "support_aware_text",
        },
        required_phases=(2, 6),
        disable_phases=(3, 4, 5, 7),
        components=("prototype", "alignment"),
    ),
    Combo(
        combo_id="C8_CUB_FINEGRAINED_OT",
        name="CUB fine-grained OT",
        purpose="Stronger fine-grained text-image alignment.",
        overrides={
            **_base_phase1("fixed", "slerp"),
            "PHASE2.ENABLED": True,
            "PHASE2.TEXT_PROTO_MODE": "combined_reweight",
            "PHASE2.VISUAL_PROTO_MODE": "shrinkage_base_prior",
            "PHASE6.ENABLED": True,
            "PHASE6.ALIGNMENT_MODE": "ot_text_image",
        },
        required_phases=(2, 6),
        disable_phases=(3, 4, 5, 7),
        components=("prototype", "alignment"),
    ),
    Combo(
        combo_id="C9_SAFE_BEST_GUESS",
        name="Safe best guess",
        purpose="Most defensible non-transductive final model.",
        overrides={
            **_base_phase1("fixed", "slerp"),
            "PHASE2.ENABLED": True,
            "PHASE2.VISUAL_PROTO_MODE": "shrinkage_base_prior",
            "PHASE2.TEXT_PROTO_MODE": "combined_reweight",
            "PHASE5.ENABLED": True,
            "PHASE5.SPACE_TRANSFORM": "common_direction_removal",
            "PHASE5.APPLY_TO": "all",
            "PHASE6.ENABLED": True,
            "PHASE6.ALIGNMENT_MODE": "support_aware_text",
            "PHASE7.ENABLED": True,
            "PHASE7.SEPARATION_MODE": "graph_highpass",
        },
        required_phases=(2, 5, 6, 7),
        disable_phases=(3, 4),
        optional_overrides=(
            ("PHASE3.DYNAMIC_ALPHA_ENABLED", False),
        ),
        components=("prototype", "space", "alignment", "separation"),
    ),
    Combo(
        combo_id="C10_AMBITIOUS_FULL",
        name="Ambitious full",
        purpose="Test whether all promising mechanisms cooperate.",
        overrides={
            **_base_phase1("fixed", "slerp"),
            "PHASE2.ENABLED": True,
            "PHASE2.VISUAL_PROTO_MODE": "shrinkage_base_prior",
            "PHASE2.TEXT_PROTO_MODE": "combined_reweight",
            "PHASE2.DYNAMIC_LAMBDA_I": True,
            "PHASE3.ENABLED": True,
            "PHASE3.HUBNESS_ENABLED": True,
            "PHASE3.DYNAMIC_ALPHA_ENABLED": True,
            "PHASE3.DYNAMIC_ALPHA_MODE": "entropy_margin",
            "PHASE4.ENABLED": True,
            "PHASE4.COV_MODE": "hybrid_diag",
            "PHASE4.APPLY_TO_NOVEL": True,
            "PHASE5.ENABLED": True,
            "PHASE5.SPACE_TRANSFORM": "common_direction_removal",
            "PHASE5.APPLY_TO": "all",
            "PHASE6.ENABLED": True,
            "PHASE6.ALIGNMENT_MODE": "support_aware_text",
            "PHASE7.ENABLED": True,
            "PHASE7.SEPARATION_MODE": "graph_highpass",
        },
        required_phases=(2, 3, 4, 5, 6, 7),
        risky=True,
        components=("prototype", "uncertainty", "covariance", "space", "alignment", "separation"),
    ),
    Combo(
        combo_id="C11_DYNAMIC_BETA_FULL",
        name="Dynamic beta full",
        purpose="Same as C10 but with dynamic beta after better prototypes.",
        overrides={
            **_base_phase1("class_margin", "slerp"),
            "PHASE2.ENABLED": True,
            "PHASE2.VISUAL_PROTO_MODE": "shrinkage_base_prior",
            "PHASE2.TEXT_PROTO_MODE": "combined_reweight",
            "PHASE2.DYNAMIC_LAMBDA_I": True,
            "PHASE3.ENABLED": True,
            "PHASE3.HUBNESS_ENABLED": True,
            "PHASE3.DYNAMIC_ALPHA_ENABLED": True,
            "PHASE3.DYNAMIC_ALPHA_MODE": "entropy_margin",
            "PHASE4.ENABLED": True,
            "PHASE4.COV_MODE": "hybrid_diag",
            "PHASE4.APPLY_TO_NOVEL": True,
            "PHASE5.ENABLED": True,
            "PHASE5.SPACE_TRANSFORM": "common_direction_removal",
            "PHASE5.APPLY_TO": "all",
            "PHASE6.ENABLED": True,
            "PHASE6.ALIGNMENT_MODE": "support_aware_text",
            "PHASE7.ENABLED": True,
            "PHASE7.SEPARATION_MODE": "graph_highpass",
        },
        required_phases=(2, 3, 4, 5, 6, 7),
        risky=True,
        components=("prototype", "uncertainty", "covariance", "space", "alignment", "separation", "dynamic_beta"),
    ),
    Combo(
        combo_id="C12_TRANSDUCTIVE_LABEL_PRIOR",
        name="Transductive label prior",
        purpose="Upper-bound style transductive correction.",
        overrides={
            **_base_phase1("fixed", "slerp"),
            "PHASE2.ENABLED": True,
            "PHASE2.VISUAL_PROTO_MODE": "shrinkage_base_prior",
            "PHASE2.TEXT_PROTO_MODE": "combined_reweight",
            "PHASE6.ENABLED": True,
            "PHASE6.ALIGNMENT_MODE": "support_aware_text",
            "PHASE6.LABEL_PRIOR_ENABLED": True,
            "PHASE6.LABEL_PRIOR_TRANSDUCTIVE": True,
        },
        required_phases=(2, 6),
        disable_phases=(3, 4, 5, 7),
        transductive=True,
        risky=True,
        components=("prototype", "alignment", "transductive_label_prior"),
    ),
)


COMBO_SETS = {
    "safe": (
        "C0_ORIGINAL",
        "C1_SAFE_PROTOTYPE",
        "C2_SAFE_PROTOTYPE_SLERP",
        "C3_SAFE_PROTOTYPE_SPACE",
        "C5_PROTOTYPE_SPACE_SYSTEM",
        "C6_NOVEL_CLASS_SYSTEM",
        "C7_CUB_FINEGRAINED_SAFE",
        "C8_CUB_FINEGRAINED_OT",
        "C9_SAFE_BEST_GUESS",
        "C10_AMBITIOUS_FULL",
    ),
    "risky": (
        "C4_UNCERTAINTY_AWARE",
        "C11_DYNAMIC_BETA_FULL",
        "C12_TRANSDUCTIVE_LABEL_PRIOR",
    ),
    "all": tuple(combo.combo_id for combo in FINAL_COMBOS),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run selected final BiMC combo ablations.")
    parser.add_argument("--datasets", nargs="+", default=["cifar100", "cub200"])
    parser.add_argument("--config", default="configs/trainers/bimc.yaml")
    parser.add_argument("--output-dir", default="results/final_combos")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu-id", default="0;")
    parser.add_argument("--combo-set", choices=sorted(COMBO_SETS), default="all")
    parser.add_argument("--continue-on-error", type=str2bool, default=True)
    parser.add_argument("--dry-run", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value.")


def get_combos():
    return list(FINAL_COMBOS)


def combo_ids_for_set(combo_set):
    return list(COMBO_SETS[combo_set])


def select_combos(combo_set):
    by_id = {combo.combo_id: combo for combo in FINAL_COMBOS}
    return [by_id[combo_id] for combo_id in COMBO_SETS[combo_set]]


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(args.output_dir, timestamp)
    os.makedirs(run_root, exist_ok=True)

    combos = select_combos(args.combo_set)
    git_info = get_git_info()
    planned_runs = build_planned_runs(args.datasets, args.config, combos)
    manifest = build_run_manifest(
        timestamp=timestamp,
        datasets=args.datasets,
        config=args.config,
        combo_set=args.combo_set,
        device=args.device,
        gpu_id=args.gpu_id,
        dry_run=args.dry_run,
        combos=combos,
        planned_runs=planned_runs,
        git_info=git_info,
    )
    write_json(os.path.join(run_root, "run_manifest.json"), manifest)

    if args.dry_run:
        print_dry_run_plan(planned_runs)
        print(f"Dry-run manifest: {os.path.join(run_root, 'run_manifest.json')}")
        return 0

    rows = []
    failed_without_continue = False
    for planned in planned_runs:
        combo = planned["combo"]
        dataset = planned["dataset"]
        combo_dir = os.path.join(run_root, dataset, combo.combo_id)
        os.makedirs(combo_dir, exist_ok=True)
        print(f"Running {dataset} / {combo.combo_id}")

        start = perf_counter()
        if planned["validation_errors"]:
            duration_sec = round(perf_counter() - start, 3)
            error_message = "; ".join(planned["validation_errors"])
            write_failed_combo_files(combo_dir, planned["config_preview"], dataset, error_message, duration_sec)
            result = {
                "status": "failed",
                "duration_sec": duration_sec,
                "output_dir": combo_dir,
                "report_path": os.path.join(combo_dir, "report.md"),
                "raw_metrics": {},
                "error_message": error_message,
            }
        else:
            result = execute_combo(planned, combo_dir, args)

        row = build_metric_row(
            dataset=dataset,
            combo=combo,
            combo_set=args.combo_set,
            result=result,
        )
        rows.append(row)
        planned["status"] = result["status"]
        planned["duration_sec"] = result["duration_sec"]
        planned["output_dir"] = combo_dir
        planned["report_path"] = result["report_path"]
        planned["error_message"] = result["error_message"]

        if result["status"] != "success" and not args.continue_on_error:
            failed_without_continue = True
            break

    rows = compute_baseline_deltas(rows)
    write_combined_outputs(run_root, rows, args, git_info, timestamp)
    manifest["planned_runs"] = serialize_planned_runs(planned_runs)
    manifest["results"] = rows
    write_json(os.path.join(run_root, "run_manifest.json"), manifest)

    print(f"Combined report: {os.path.join(run_root, 'combined_final_combo_report.md')}")
    return 1 if failed_without_continue else 0


def build_planned_runs(datasets, config_path, combos):
    plans = []
    train_cfg_path = resolve_train_cfg(config_path)
    train_config = load_config(train_cfg_path)

    for dataset in datasets:
        data_cfg_path = resolve_dataset_cfg(dataset)
        config_preview = deep_update(load_config(data_cfg_path), train_config)
        for combo in combos:
            resolved = resolve_combo_overrides(combo, config_preview)
            validation_errors = validate_combo_support(combo, config_preview, resolved)
            preview = apply_nested_overrides(config_preview, resolved["overrides"])
            plans.append({
                "dataset": dataset,
                "dataset_config": data_cfg_path,
                "train_config": train_cfg_path,
                "combo": combo,
                "abstract_overrides": deepcopy(combo.overrides),
                "resolved_overrides": resolved["resolved_paths"],
                "mapped_overrides": resolved["abstract_to_resolved"],
                "optional_skipped": resolved["optional_skipped"],
                "validation_errors": validation_errors,
                "config_preview": preview,
            })
    return plans


def execute_combo(planned, combo_dir, args):
    try:
        from main import setup_cfg
    except Exception as exc:
        duration_sec = 0.0
        error_message = f"Could not import evaluation pipeline: {exc}"
        write_failed_combo_files(combo_dir, planned["config_preview"], planned["dataset"], error_message, duration_sec)
        return {
            "status": "failed",
            "duration_sec": duration_sec,
            "output_dir": combo_dir,
            "report_path": os.path.join(combo_dir, "report.md"),
            "raw_metrics": {},
            "error_message": error_message,
        }

    cfg = setup_cfg(planned["dataset_config"], planned["train_config"])
    apply_yacs_overrides(cfg, planned["resolved_overrides"])
    cfg.defrost()
    cfg.RUN_NAME = planned["combo"].combo_id
    if args.seed is not None:
        cfg.SEED = args.seed
    cfg.freeze()
    return run_eval_with_config(
        cfg,
        combo_dir,
        planned["dataset"],
        args.device,
        args.gpu_id,
        seed=args.seed,
    )


def resolve_combo_overrides(combo, config_dict):
    resolved_paths = {}
    abstract_to_resolved = {}
    missing = []
    optional_skipped = []

    for phase in combo.disable_phases:
        if not phase_config_available(phase, config_dict):
            continue
        abstract_path = f"PHASE{phase}.ENABLED"
        actual_path = resolve_override_path(abstract_path, config_dict)
        if actual_path is None:
            missing.append(abstract_path)
            continue
        abstract_to_resolved[abstract_path] = actual_path
        resolved_paths[actual_path] = False

    for abstract_path, value in combo.overrides.items():
        actual_path = resolve_override_path(abstract_path, config_dict)
        if actual_path is None:
            missing.append(abstract_path)
            continue
        abstract_to_resolved[abstract_path] = actual_path
        resolved_paths[actual_path] = value

    for abstract_path, value in combo.optional_overrides:
        phase = phase_number_from_path(abstract_path)
        if phase is not None and not phase_available(phase, config_dict):
            optional_skipped.append({
                "override": abstract_path,
                "reason": f"Phase {phase} is not implemented/configured in this checkout.",
            })
            continue
        actual_path = resolve_override_path(abstract_path, config_dict)
        if actual_path is None:
            missing.append(abstract_path)
            continue
        abstract_to_resolved[abstract_path] = actual_path
        resolved_paths[actual_path] = value

    return {
        "resolved_paths": resolved_paths,
        "abstract_to_resolved": abstract_to_resolved,
        "missing": missing,
        "optional_skipped": optional_skipped,
        "overrides": nested_from_dotted(resolved_paths),
    }


def validate_combo_support(combo, config_dict, resolved):
    errors = []
    for phase in combo.required_phases:
        info = PHASES[phase]
        missing_modules = [module for module in info["modules"] if importlib.util.find_spec(module) is None]
        if missing_modules:
            errors.append(
                f"{info['label']} implementation missing: expected module(s) {', '.join(missing_modules)}."
            )
        if not phase_config_available(phase, config_dict):
            phase_name = f"PHASE{phase}"
            errors.append(
                f"{info['label']} config missing: expected {phase_name}, "
                f"TRAINER.BiMC.{phase_name}, or TRAINER.BiMC.{phase_name}_* keys."
            )

    if resolved["missing"]:
        errors.append(
            "Could not map override key(s) to the current config: "
            + ", ".join(resolved["missing"])
            + "."
        )
    return errors


def phase_available(phase, config_dict):
    info = PHASES.get(phase)
    if not info:
        return False
    modules_available = all(importlib.util.find_spec(module) is not None for module in info["modules"])
    return modules_available and phase_config_available(phase, config_dict)


def phase_config_available(phase, config_dict):
    phase_name = f"PHASE{phase}"
    if path_exists(config_dict, phase_name):
        return True
    if path_exists(config_dict, f"TRAINER.BiMC.{phase_name}"):
        return True
    trainer = get_path(config_dict, "TRAINER.BiMC")
    return isinstance(trainer, dict) and any(str(key).startswith(f"{phase_name}_") for key in trainer)


def resolve_override_path(abstract_path, config_dict):
    candidates = override_candidates(abstract_path)
    for candidate in candidates:
        if path_exists(config_dict, candidate):
            return candidate
    return None


def override_candidates(abstract_path):
    parts = abstract_path.split(".")
    if len(parts) == 1:
        return [abstract_path]

    phase = parts[0].upper()
    key = ".".join(parts[1:]).upper()
    flat_key = key.replace(".", "_")

    if phase == "PHASE1":
        return [
            f"TRAINER.BiMC.{flat_key}",
            f"PHASE1.{key}",
            f"MODEL.{flat_key}",
            flat_key,
        ]

    if phase.startswith("PHASE"):
        return [
            f"{phase}.{key}",
            f"TRAINER.BiMC.{phase}.{key}",
            f"TRAINER.BiMC.{phase}_{flat_key}",
            f"MODEL.{phase}.{key}",
            f"MODEL.{phase}_{flat_key}",
        ]
    return [abstract_path]


def phase_number_from_path(path):
    first = path.split(".", 1)[0].upper()
    if first.startswith("PHASE"):
        try:
            return int(first.replace("PHASE", ""))
        except ValueError:
            return None
    return None


def apply_nested_overrides(config_dict, nested_overrides):
    return deep_update(config_dict, nested_overrides)


def apply_yacs_overrides(cfg, resolved_paths):
    cfg.defrost()
    for dotted_path, value in resolved_paths.items():
        set_cfg_path(cfg, dotted_path, value)
    cfg.freeze()


def set_cfg_path(cfg, dotted_path, value):
    node = cfg
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        node = getattr(node, part)
    setattr(node, parts[-1], value)


def nested_from_dotted(dotted_values):
    result = {}
    for dotted_path, value in dotted_values.items():
        node = result
        parts = dotted_path.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return result


def path_exists(config_dict, dotted_path):
    sentinel = object()
    return get_path(config_dict, dotted_path, sentinel) is not sentinel


def get_path(config_dict, dotted_path, default=None):
    node = config_dict
    for part in dotted_path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def build_metric_row(dataset, combo, combo_set, result):
    metrics = normalize_metric_row(result.get("raw_metrics"))
    return {
        "dataset": dataset,
        "combo_id": combo.combo_id,
        "combo_name": combo.name,
        "combo_set": combo_set,
        "transductive": combo.transductive,
        "risky": combo.risky,
        **metrics,
        "delta_avg_vs_original": None,
        "delta_final_vs_original": None,
        "delta_pd_vs_original": None,
        "status": result.get("status"),
        "duration_sec": result.get("duration_sec"),
        "output_dir": result.get("output_dir"),
        "report_path": result.get("report_path"),
        "error_message": result.get("error_message"),
    }


def compute_baseline_deltas(rows):
    rows = [dict(row) for row in rows]
    by_dataset = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], []).append(row)

    for dataset_rows in by_dataset.values():
        baseline = next((row for row in dataset_rows if row["combo_id"] == "C0_ORIGINAL"), None)
        for row in dataset_rows:
            row["delta_avg_vs_original"] = delta_or_none(row.get("average_accuracy"), baseline, "average_accuracy")
            row["delta_final_vs_original"] = delta_or_none(row.get("final_accuracy"), baseline, "final_accuracy")
            row["delta_pd_vs_original"] = delta_or_none(row.get("performance_degradation"), baseline, "performance_degradation")
    return rows


def delta_or_none(value, baseline, key):
    if value is None or not baseline or baseline.get(key) is None:
        return None
    return round(float(value) - float(baseline[key]), 4)


def write_combined_outputs(run_root, rows, args, git_info, timestamp):
    csv_path = os.path.join(run_root, "combined_final_combo_metrics.csv")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in CSV_COLUMNS})

    write_json(os.path.join(run_root, "combined_final_combo_metrics.json"), rows)
    report = render_combined_report(rows, args, git_info, timestamp)
    with open(os.path.join(run_root, "combined_final_combo_report.md"), "w") as handle:
        handle.write(report)


def render_combined_report(rows, args, git_info, timestamp):
    lines = [
        "# BiMC Final Combo Report",
        "",
        "## Run Info",
        f"- timestamp: {timestamp}",
        f"- git branch: {git_info.get('branch') or 'unknown'}",
        f"- git commit: {git_info.get('commit') or 'unknown'}",
        f"- datasets: {', '.join(args.datasets)}",
        f"- config: {args.config}",
        f"- combo set: {args.combo_set}",
        f"- device: {args.device}",
        f"- gpu id: {args.gpu_id}",
        "",
        "## Execution Summary",
        "",
        "| dataset | combo | status | duration | transductive | risky |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['dataset']} | {row['combo_id']} | {row['status']} | "
            f"{fmt(row.get('duration_sec'))} | {row['transductive']} | {row['risky']} |"
        )

    lines.extend(["", "## Best Results by Dataset", ""])
    for dataset in sorted({row["dataset"] for row in rows}):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        lines.extend(render_best_results(dataset, dataset_rows))

    lines.extend([
        "",
        "## Full Comparison",
        "",
        "| dataset | combo | avg | final | pd | delta_avg | delta_final | transductive | risky | status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ])
    for row in rows:
        lines.append(
            "| "
            f"{row['dataset']} | {row['combo_id']} | {fmt(row.get('average_accuracy'))} | "
            f"{fmt(row.get('final_accuracy'))} | {fmt(row.get('performance_degradation'))} | "
            f"{fmt(row.get('delta_avg_vs_original'))} | {fmt(row.get('delta_final_vs_original'))} | "
            f"{row['transductive']} | {row['risky']} | {row['status']} |"
        )

    lines.extend(["", "## Interpretation", ""])
    lines.extend(render_interpretation(rows))
    lines.append("")
    return "\n".join(lines)


def render_best_results(dataset, rows):
    lines = [f"### {dataset}", ""]
    baseline = next((row for row in rows if row["combo_id"] == "C0_ORIGINAL"), None)
    lines.append(f"- original baseline: {describe_row(baseline)}")
    lines.append(f"- best safe combo by average accuracy: {describe_best(rows, 'average_accuracy', safe=True)}")
    lines.append(f"- best safe combo by final accuracy: {describe_best(rows, 'final_accuracy', safe=True)}")
    lines.append(f"- best risky combo by average accuracy: {describe_best(rows, 'average_accuracy', risky=True)}")
    lines.append(f"- best risky combo by final accuracy: {describe_best(rows, 'final_accuracy', risky=True)}")
    lines.append(f"- best transductive combo separately: {describe_best(rows, 'average_accuracy', transductive=True)}")
    lines.append("")
    return lines


def describe_best(rows, metric, safe=False, risky=False, transductive=False):
    candidates = [
        row for row in rows
        if row.get("status") == "success" and row.get(metric) is not None
    ]
    if safe:
        candidates = [row for row in candidates if not row.get("risky") and not row.get("transductive")]
    if risky:
        candidates = [row for row in candidates if row.get("risky") and not row.get("transductive")]
    if transductive:
        candidates = [row for row in candidates if row.get("transductive")]
    if not candidates:
        return "unavailable"
    best = max(candidates, key=lambda row: row[metric])
    delta_key = "delta_avg_vs_original" if metric == "average_accuracy" else "delta_final_vs_original"
    return f"{best['combo_id']} ({fmt(best[metric])}, delta {fmt(best.get(delta_key))})"


def describe_row(row):
    if not row or row.get("status") != "success":
        return "unavailable"
    return f"{row['combo_id']} (avg {fmt(row.get('average_accuracy'))}, final {fmt(row.get('final_accuracy'))})"


def render_interpretation(rows):
    lines = []
    safe_improved = [
        row for row in rows
        if row.get("status") == "success"
        and not row.get("risky")
        and not row.get("transductive")
        and ((row.get("delta_avg_vs_original") or 0) > 0 or (row.get("delta_final_vs_original") or 0) > 0)
    ]
    if safe_improved:
        lines.append("- At least one safe combo improved over original.")
        best = max(safe_improved, key=lambda row: row.get("delta_avg_vs_original") or float("-inf"))
        combo = combo_by_id(best["combo_id"])
        source = ", ".join(component for component in combo.components if component != "original") or "unknown"
        lines.append(f"- The strongest observed safe improvement is associated with: {source}.")
    else:
        lines.append("- No safe combo improved over original, or metrics are unavailable.")
        lines.append("- The dominant improvement source cannot be identified from the available metrics.")

    dynamic_pairs = compare_dynamic_beta(rows)
    if dynamic_pairs:
        harmful = any(delta is not None and delta < 0 for delta in dynamic_pairs)
        lines.append(f"- Dynamic beta is {'still harmful' if harmful else 'not clearly harmful'} in completed paired comparisons.")
    else:
        lines.append("- Dynamic beta impact needs completed C10/C11 or C0/C4 comparisons.")

    if any(row.get("transductive") for row in rows):
        lines.append("- Transductive label-prior correction is marked transductive and should be reported separately.")

    keep = choose_keep_combo(rows)
    reject = choose_reject_combos(rows)
    lines.append(f"- Keep for next experiments: {keep}.")
    lines.append(f"- Reject or deprioritize: {reject}.")

    suspicious = suspicious_drops(rows)
    if suspicious:
        lines.append("- Suspicious drops: " + "; ".join(suspicious))
    else:
        lines.append("- Suspicious drops: none detected from available metrics.")
    return lines


def compare_dynamic_beta(rows):
    deltas = []
    by_dataset = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], {})[row["combo_id"]] = row
    for dataset_rows in by_dataset.values():
        full = dataset_rows.get("C10_AMBITIOUS_FULL")
        dynamic = dataset_rows.get("C11_DYNAMIC_BETA_FULL")
        if full and dynamic and full.get("final_accuracy") is not None and dynamic.get("final_accuracy") is not None:
            deltas.append(dynamic["final_accuracy"] - full["final_accuracy"])
        baseline = dataset_rows.get("C0_ORIGINAL")
        uncertainty = dataset_rows.get("C4_UNCERTAINTY_AWARE")
        if baseline and uncertainty and baseline.get("final_accuracy") is not None and uncertainty.get("final_accuracy") is not None:
            deltas.append(uncertainty["final_accuracy"] - baseline["final_accuracy"])
    return deltas


def choose_keep_combo(rows):
    candidates = [
        row for row in rows
        if row.get("status") == "success"
        and not row.get("transductive")
        and row.get("average_accuracy") is not None
    ]
    if not candidates:
        return "needs completed metrics"
    best = max(candidates, key=lambda row: row["average_accuracy"])
    return best["combo_id"]


def choose_reject_combos(rows):
    rejected = [
        row["combo_id"] for row in rows
        if row.get("status") == "success"
        and (
            (row.get("delta_final_vs_original") is not None and row["delta_final_vs_original"] < 0)
            or row.get("final_accuracy") is not None and row["final_accuracy"] <= 2.0
        )
    ]
    if not rejected:
        failed = sorted({row["combo_id"] for row in rows if row.get("status") == "failed"})
        return ", ".join(failed) if failed else "none from available metrics"
    return ", ".join(sorted(set(rejected)))


def suspicious_drops(rows):
    messages = []
    for row in rows:
        final_acc = row.get("final_accuracy")
        delta_final = row.get("delta_final_vs_original")
        if delta_final is not None and delta_final < -10:
            messages.append(f"{row['dataset']} {row['combo_id']} final delta {fmt(delta_final)}")
        elif final_acc is not None and final_acc <= 2.0:
            messages.append(f"{row['dataset']} {row['combo_id']} near-random final {fmt(final_acc)}")
    return messages


def combo_by_id(combo_id):
    return next(combo for combo in FINAL_COMBOS if combo.combo_id == combo_id)


def build_run_manifest(timestamp, datasets, config, combo_set, device, gpu_id, dry_run, combos, planned_runs, git_info=None):
    return {
        "timestamp": timestamp,
        "git": git_info or {},
        "datasets": list(datasets),
        "config": config,
        "combo_set": combo_set,
        "device": device,
        "gpu_id": gpu_id,
        "dry_run": dry_run,
        "combos": [
            {
                "combo_id": combo.combo_id,
                "combo_name": combo.name,
                "purpose": combo.purpose,
                "transductive": combo.transductive,
                "risky": combo.risky,
                "required_phases": list(combo.required_phases),
                "disable_phases": list(combo.disable_phases),
                "components": list(combo.components),
            }
            for combo in combos
        ],
        "planned_runs": serialize_planned_runs(planned_runs),
    }


def serialize_planned_runs(planned_runs):
    serialized = []
    for plan in planned_runs:
        combo = plan["combo"]
        serialized.append({
            "dataset": plan["dataset"],
            "combo_id": combo.combo_id,
            "combo_name": combo.name,
            "transductive": combo.transductive,
            "risky": combo.risky,
            "abstract_overrides": plan.get("abstract_overrides", {}),
            "resolved_overrides": plan.get("resolved_overrides", {}),
            "mapped_overrides": plan.get("mapped_overrides", {}),
            "optional_skipped": plan.get("optional_skipped", []),
            "validation_errors": plan.get("validation_errors", []),
            "status": plan.get("status", "planned"),
            "duration_sec": plan.get("duration_sec"),
            "output_dir": plan.get("output_dir"),
            "report_path": plan.get("report_path"),
            "error_message": plan.get("error_message"),
        })
    return serialized


def print_dry_run_plan(planned_runs):
    for plan in planned_runs:
        combo = plan["combo"]
        print(f"{plan['dataset']} / {combo.combo_id} ({combo.name})")
        for key, value in plan["abstract_overrides"].items():
            mapped = plan.get("mapped_overrides", {}).get(key, "unresolved")
            print(f"  {key} = {value} -> {mapped}")
        if plan["optional_skipped"]:
            for skipped in plan["optional_skipped"]:
                print(f"  optional skipped: {skipped['override']} ({skipped['reason']})")
        if plan["validation_errors"]:
            print("  validation errors:")
            for error in plan["validation_errors"]:
                print(f"    - {error}")


def write_failed_combo_files(output_dir, config_preview, dataset, error_message, duration_sec):
    os.makedirs(output_dir, exist_ok=True)
    save_config_snapshot(config_preview, os.path.join(output_dir, "config.json"))
    write_json(os.path.join(output_dir, "metrics.json"), {})
    with open(os.path.join(output_dir, "session_accuracy.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["session", "mean_acc", "base_avg_acc", "inc_avg_acc", "harmonic_acc", "task_acc"])
    with open(os.path.join(output_dir, "stdout.log"), "w") as handle:
        handle.write("")
    with open(os.path.join(output_dir, "stderr.log"), "w") as handle:
        handle.write(error_message + "\n")
    with open(os.path.join(output_dir, "report.md"), "w") as handle:
        handle.write(
            "\n".join([
                "# Final Combo Run",
                "",
                f"- Dataset: {dataset}",
                "- Status: failed",
                f"- Duration sec: {duration_sec:.3f}",
                "",
                "## Error",
                "",
                error_message,
                "",
            ])
        )


def get_git_info():
    return {
        "branch": run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": run_git(["rev-parse", "HEAD"]),
    }


def run_git(args):
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def write_json(path, value):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(to_plain(value), handle, indent=2)


def fmt(value):
    if value is None or value == "":
        return "n/a"
    return f"{float(value):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
