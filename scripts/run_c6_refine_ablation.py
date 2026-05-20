import argparse
import csv
import os
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
)
from scripts.run_final_combo_ablation import (
    apply_yacs_overrides,
    fmt,
    get_git_info,
    resolve_combo_overrides,
    validate_combo_support,
    write_json,
)


CSV_COLUMNS = [
    "dataset",
    "variant_id",
    "variant_name",
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
    "error_message",
]


@dataclass(frozen=True)
class Variant:
    variant_id: str
    name: str
    purpose: str
    overrides: dict
    required_phases: tuple = ()
    disable_phases: tuple = ()
    optional_overrides: tuple = ()
    notes: tuple = ()


def _base_phase1(beta_mode="fixed", geometry="linear", reliability_mode="entropy_margin"):
    return {
        "PHASE1.FUSION_BETA_MODE": beta_mode,
        "PHASE1.FUSION_GEOMETRY": geometry,
        "PHASE1.RELIABILITY_MODE": reliability_mode,
    }


C6_ORIGINAL_OVERRIDES = {
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
}


def _c6_overrides(**updates):
    overrides = dict(C6_ORIGINAL_OVERRIDES)
    overrides.update(updates)
    return overrides


C6_VARIANTS = (
    Variant(
        variant_id="R0_ORIGINAL",
        name="Original BiMC",
        purpose="Original BiMC with all phase modifications disabled.",
        overrides=_base_phase1("fixed", "linear"),
        disable_phases=(2, 3, 4, 5, 6, 7),
    ),
    Variant(
        variant_id="R1_C6_ORIGINAL",
        name="Current C6",
        purpose="Current C6_NOVEL_CLASS_SYSTEM settings.",
        overrides=C6_ORIGINAL_OVERRIDES,
        required_phases=(2, 3, 4),
        disable_phases=(5, 6, 7),
    ),
    Variant(
        variant_id="R2_C6_NOVEL_ONLY",
        name="C6 novel only",
        purpose="Keep base branch on original fixed-alpha scoring while applying C6 to novel classes.",
        overrides=_c6_overrides(**{"PHASE3.DYNAMIC_ALPHA_NOVEL_ONLY": True}),
        required_phases=(2, 3, 4),
        disable_phases=(5, 6, 7),
        notes=(
            "Phase4 APPLY_TO_NOVEL keeps base auxiliary scoring on the original covariance path.",
            "This variant additionally prevents dynamic alpha from changing the base branch.",
        ),
    ),
    Variant(
        variant_id="R3_C6_NO_DYNAMIC_LAMBDA",
        name="C6 no dynamic lambda",
        purpose="Disable Phase 2 per-class dynamic visual shrinkage.",
        overrides=_c6_overrides(**{"PHASE2.DYNAMIC_LAMBDA_I": False}),
        required_phases=(2, 3, 4),
        disable_phases=(5, 6, 7),
    ),
    Variant(
        variant_id="R4_C6_LOW_LAMBDA_CAP_02",
        name="C6 lambda cap 0.2",
        purpose="Limit dynamic visual shrinkage to 0.2.",
        overrides=_c6_overrides(
            **{
                "PHASE2.DYNAMIC_LAMBDA_I": True,
                "PHASE2.DYNAMIC_LAMBDA_MAX": 0.2,
            }
        ),
        required_phases=(2, 3, 4),
        disable_phases=(5, 6, 7),
    ),
    Variant(
        variant_id="R5_C6_LOW_LAMBDA_CAP_03",
        name="C6 lambda cap 0.3",
        purpose="Limit dynamic visual shrinkage to 0.3.",
        overrides=_c6_overrides(
            **{
                "PHASE2.DYNAMIC_LAMBDA_I": True,
                "PHASE2.DYNAMIC_LAMBDA_MAX": 0.3,
            }
        ),
        required_phases=(2, 3, 4),
        disable_phases=(5, 6, 7),
    ),
    Variant(
        variant_id="R6_C6_NO_SHRINKAGE",
        name="C6 no shrinkage",
        purpose="Use the raw support mean visual prototype while keeping dynamic lambda enabled.",
        overrides=_c6_overrides(
            **{
                "PHASE2.VISUAL_PROTO_MODE": "mean",
                "PHASE2.DYNAMIC_LAMBDA_I": True,
            }
        ),
        required_phases=(2, 3, 4),
        disable_phases=(5, 6, 7),
    ),
    Variant(
        variant_id="R7_C6_COV_WEIGHT_010",
        name="C6 cov weight 0.10",
        purpose="Blend KNN and covariance novel auxiliaries with fixed covariance weight 0.10.",
        overrides=_c6_overrides(
            **{
                "PHASE4.COV_SCORE_WEIGHT": 0.10,
                "PHASE4.NOVEL_AUX_COMBINE_MODE": "weighted",
            }
        ),
        required_phases=(2, 3, 4),
        disable_phases=(5, 6, 7),
    ),
    Variant(
        variant_id="R8_C6_COV_WEIGHT_025",
        name="C6 cov weight 0.25",
        purpose="Blend KNN and covariance novel auxiliaries with fixed covariance weight 0.25.",
        overrides=_c6_overrides(
            **{
                "PHASE4.COV_SCORE_WEIGHT": 0.25,
                "PHASE4.NOVEL_AUX_COMBINE_MODE": "weighted",
            }
        ),
        required_phases=(2, 3, 4),
        disable_phases=(5, 6, 7),
    ),
    Variant(
        variant_id="R9_C6_COV_WEIGHT_040",
        name="C6 cov weight 0.40",
        purpose="Blend KNN and covariance novel auxiliaries with fixed covariance weight 0.40.",
        overrides=_c6_overrides(
            **{
                "PHASE4.COV_SCORE_WEIGHT": 0.40,
                "PHASE4.NOVEL_AUX_COMBINE_MODE": "weighted",
            }
        ),
        required_phases=(2, 3, 4),
        disable_phases=(5, 6, 7),
    ),
    Variant(
        variant_id="R10_C6_RELIABILITY_GATED_COV",
        name="C6 reliability gated cov",
        purpose="Gate the covariance novel auxiliary by entropy-margin reliability.",
        overrides=_c6_overrides(
            **{
                "PHASE4.NOVEL_AUX_COMBINE_MODE": "reliability_gated",
            }
        ),
        required_phases=(2, 3, 4),
        disable_phases=(5, 6, 7),
    ),
    Variant(
        variant_id="R11_C6_DYNAMIC_ALPHA_NOVEL_ONLY",
        name="C6 dynamic alpha novel only",
        purpose="Apply dynamic alpha only to the novel branch.",
        overrides=_c6_overrides(**{"PHASE3.DYNAMIC_ALPHA_NOVEL_ONLY": True}),
        required_phases=(2, 3, 4),
        disable_phases=(5, 6, 7),
    ),
    Variant(
        variant_id="R12_C6_ALPHA_CLIP_SAFE",
        name="C6 safe alpha clip",
        purpose="Clip dynamic alpha to [0.40, 0.90].",
        overrides=_c6_overrides(
            **{
                "PHASE3.ALPHA_CLIP_MIN": 0.40,
                "PHASE3.ALPHA_CLIP_MAX": 0.90,
            }
        ),
        required_phases=(2, 3, 4),
        disable_phases=(5, 6, 7),
    ),
    Variant(
        variant_id="R13_C6_FIXED_ALPHA",
        name="C6 fixed alpha",
        purpose="Use original fixed alpha instead of Phase 3 dynamic alpha.",
        overrides=_c6_overrides(**{"PHASE3.DYNAMIC_ALPHA_ENABLED": False}),
        required_phases=(2, 3, 4),
        disable_phases=(5, 6, 7),
    ),
    Variant(
        variant_id="R14_C6_SLERP",
        name="C6 SLERP",
        purpose="Keep fixed beta and use SLERP prototype fusion geometry.",
        overrides=_c6_overrides(**_base_phase1("fixed", "slerp")),
        required_phases=(2, 3, 4),
        disable_phases=(5, 6, 7),
    ),
    Variant(
        variant_id="R15_C6_BEST_SAFE",
        name="C6 best safe",
        purpose="Combine conservative C6 settings expected to reduce CIFAR regressions.",
        overrides=_c6_overrides(
            **{
                **_base_phase1("fixed", "slerp"),
                "PHASE2.DYNAMIC_LAMBDA_I": True,
                "PHASE2.DYNAMIC_LAMBDA_MAX": 0.3,
                "PHASE4.NOVEL_AUX_COMBINE_MODE": "reliability_gated",
                "PHASE4.COV_SCORE_WEIGHT": 0.25,
                "PHASE3.DYNAMIC_ALPHA_NOVEL_ONLY": True,
            }
        ),
        required_phases=(2, 3, 4),
        disable_phases=(5, 6, 7),
    ),
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run C6 refinement ablations.")
    parser.add_argument("--datasets", nargs="+", default=["cifar100", "cub200"])
    parser.add_argument("--config", default="configs/trainers/bimc.yaml")
    parser.add_argument("--output-dir", default="results/c6_refine")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu-id", default="0;")
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


def get_variants():
    return list(C6_VARIANTS)


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(args.output_dir, timestamp)
    os.makedirs(run_root, exist_ok=True)

    variants = get_variants()
    git_info = get_git_info()
    planned_runs = build_planned_runs(args.datasets, args.config, variants)
    manifest = build_run_manifest(
        timestamp=timestamp,
        datasets=args.datasets,
        config=args.config,
        device=args.device,
        gpu_id=args.gpu_id,
        dry_run=args.dry_run,
        seed=args.seed,
        variants=variants,
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
        variant = planned["variant"]
        dataset = planned["dataset"]
        variant_dir = os.path.join(run_root, dataset, variant.variant_id)
        os.makedirs(variant_dir, exist_ok=True)
        print(f"Running {dataset} / {variant.variant_id}")

        start = perf_counter()
        if planned["validation_errors"]:
            duration_sec = round(perf_counter() - start, 3)
            error_message = "; ".join(planned["validation_errors"])
            write_failed_variant_files(
                variant_dir,
                planned["config_preview"],
                dataset,
                variant,
                error_message,
                duration_sec,
            )
            result = {
                "status": "failed",
                "duration_sec": duration_sec,
                "output_dir": variant_dir,
                "raw_metrics": {},
                "error_message": error_message,
            }
        else:
            result = execute_variant(planned, variant_dir, args)
            write_variant_report(variant_dir, dataset, variant, result)

        row = build_metric_row(dataset=dataset, variant=variant, result=result)
        rows.append(row)
        planned["status"] = result["status"]
        planned["duration_sec"] = result["duration_sec"]
        planned["output_dir"] = variant_dir
        planned["error_message"] = result["error_message"]

        if result["status"] != "success" and not args.continue_on_error:
            failed_without_continue = True
            break

    rows = compute_baseline_deltas(rows)
    write_combined_outputs(run_root, rows, args, git_info, timestamp)
    manifest["planned_runs"] = serialize_planned_runs(planned_runs)
    manifest["results"] = rows
    write_json(os.path.join(run_root, "run_manifest.json"), manifest)

    print(f"Combined report: {os.path.join(run_root, 'combined_c6_refine_report.md')}")
    return 1 if failed_without_continue else 0


def build_planned_runs(datasets, config_path, variants):
    plans = []
    train_cfg_path = resolve_train_cfg(config_path)
    train_config = load_config(train_cfg_path)

    for dataset in datasets:
        data_cfg_path = resolve_dataset_cfg(dataset)
        config_preview = deep_update(load_config(data_cfg_path), train_config)
        for variant in variants:
            resolved = resolve_combo_overrides(variant, config_preview)
            validation_errors = validate_combo_support(variant, config_preview, resolved)
            preview = deep_update(config_preview, resolved["overrides"])
            plans.append(
                {
                    "dataset": dataset,
                    "dataset_config": data_cfg_path,
                    "train_config": train_cfg_path,
                    "variant": variant,
                    "abstract_overrides": deepcopy(variant.overrides),
                    "resolved_overrides": resolved["resolved_paths"],
                    "mapped_overrides": resolved["abstract_to_resolved"],
                    "optional_skipped": resolved["optional_skipped"],
                    "validation_errors": validation_errors,
                    "config_preview": preview,
                }
            )
    return plans


def execute_variant(planned, variant_dir, args):
    try:
        from main import setup_cfg
    except Exception as exc:
        duration_sec = 0.0
        error_message = f"Could not import evaluation pipeline: {exc}"
        write_failed_variant_files(
            variant_dir,
            planned["config_preview"],
            planned["dataset"],
            planned["variant"],
            error_message,
            duration_sec,
        )
        return {
            "status": "failed",
            "duration_sec": duration_sec,
            "output_dir": variant_dir,
            "raw_metrics": {},
            "error_message": error_message,
        }

    cfg = setup_cfg(planned["dataset_config"], planned["train_config"])
    apply_yacs_overrides(cfg, planned["resolved_overrides"])
    cfg.defrost()
    cfg.RUN_NAME = planned["variant"].variant_id
    if args.seed is not None:
        cfg.SEED = args.seed
    cfg.freeze()
    return run_eval_with_config(
        cfg,
        variant_dir,
        planned["dataset"],
        args.device,
        args.gpu_id,
        seed=args.seed,
    )


def build_metric_row(dataset, variant, result):
    metrics = normalize_metric_row(result.get("raw_metrics"))
    return {
        "dataset": dataset,
        "variant_id": variant.variant_id,
        "variant_name": variant.name,
        **metrics,
        "delta_avg_vs_original": None,
        "delta_final_vs_original": None,
        "delta_pd_vs_original": None,
        "status": result.get("status"),
        "duration_sec": result.get("duration_sec"),
        "output_dir": result.get("output_dir"),
        "error_message": result.get("error_message"),
    }


def compute_baseline_deltas(rows):
    rows = [dict(row) for row in rows]
    by_dataset = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], []).append(row)

    for dataset_rows in by_dataset.values():
        baseline = next((row for row in dataset_rows if row["variant_id"] == "R0_ORIGINAL"), None)
        for row in dataset_rows:
            row["delta_avg_vs_original"] = delta_or_none(row.get("average_accuracy"), baseline, "average_accuracy")
            row["delta_final_vs_original"] = delta_or_none(row.get("final_accuracy"), baseline, "final_accuracy")
            row["delta_pd_vs_original"] = delta_or_none(
                row.get("performance_degradation"),
                baseline,
                "performance_degradation",
            )
    return rows


def delta_or_none(value, baseline, key):
    if value is None or not baseline or baseline.get(key) is None:
        return None
    return round(float(value) - float(baseline[key]), 4)


def write_combined_outputs(run_root, rows, args, git_info, timestamp):
    csv_path = os.path.join(run_root, "combined_c6_refine_metrics.csv")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in CSV_COLUMNS})

    write_json(os.path.join(run_root, "combined_c6_refine_metrics.json"), rows)
    report = render_combined_report(rows, args, git_info, timestamp)
    with open(os.path.join(run_root, "combined_c6_refine_report.md"), "w") as handle:
        handle.write(report)


def render_combined_report(rows, args, git_info, timestamp):
    lines = [
        "# C6 Refinement Ablation Report",
        "",
        "## Run Info",
        f"- timestamp: {timestamp}",
        f"- git branch: {git_info.get('branch') or 'unknown'}",
        f"- git commit: {git_info.get('commit') or 'unknown'}",
        f"- datasets: {', '.join(args.datasets)}",
        f"- config: {args.config}",
        f"- device: {args.device}",
        f"- gpu id: {args.gpu_id}",
        f"- seed: {args.seed if args.seed is not None else 'config default'}",
        "",
        "## C6 Base-Branch Scope",
        "- Phase4 APPLY_TO_NOVEL keeps base auxiliary scoring on the original covariance path in this checkout.",
        "- R2, R11, and R15 additionally set PHASE3.DYNAMIC_ALPHA_NOVEL_ONLY so base classes use fixed alpha.",
        "",
        "## Dataset Findings",
        "",
    ]
    for dataset in sorted({row["dataset"] for row in rows}):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        lines.extend(render_dataset_summary(dataset, dataset_rows))

    lines.extend(
        [
            "## Cross-Dataset Interpretation",
            "",
            f"- whether C6 improves CUB: {c6_improvement_statement(rows, 'cub')}",
            f"- whether C6 still hurts CIFAR: {c6_hurt_statement(rows, 'cifar')}",
            f"- recommendation: {recommendation(rows)}",
            "",
            "## Suspicious Drops",
            "",
        ]
    )
    suspicious = suspicious_drops(rows)
    if suspicious:
        lines.extend(f"- {item}" for item in suspicious)
    else:
        lines.append("- none detected from available metrics")

    lines.extend(
        [
            "",
            "## Full Comparison",
            "",
            "| dataset | variant | avg | final | pd | base final | novel final | delta avg | delta final | status |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"{row['dataset']} | {row['variant_id']} | {fmt(row.get('average_accuracy'))} | "
            f"{fmt(row.get('final_accuracy'))} | {fmt(row.get('performance_degradation'))} | "
            f"{fmt(row.get('base_final_accuracy'))} | {fmt(row.get('novel_final_accuracy'))} | "
            f"{fmt(row.get('delta_avg_vs_original'))} | {fmt(row.get('delta_final_vs_original'))} | "
            f"{row['status']} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_dataset_summary(dataset, rows):
    lines = [f"### {dataset}", ""]
    baseline = row_by_variant(rows, "R0_ORIGINAL")
    current_c6 = row_by_variant(rows, "R1_C6_ORIGINAL")
    lines.append(f"- original baseline: {describe_row(baseline)}")
    lines.append(f"- current C6 result: {describe_row(current_c6)}")
    lines.append(f"- best variant by average accuracy: {describe_best(rows, 'average_accuracy')}")
    lines.append(f"- best variant by final accuracy: {describe_best(rows, 'final_accuracy')}")
    lines.append(f"- best variant by novel accuracy: {describe_best(rows, 'novel_final_accuracy')}")
    lines.append("")
    return lines


def describe_row(row):
    if not row:
        return "unavailable"
    if row.get("status") != "success":
        return f"{row['variant_id']} failed"
    return (
        f"{row['variant_id']} "
        f"(avg {fmt(row.get('average_accuracy'))}, final {fmt(row.get('final_accuracy'))}, "
        f"delta avg {fmt(row.get('delta_avg_vs_original'))}, "
        f"delta final {fmt(row.get('delta_final_vs_original'))})"
    )


def describe_best(rows, metric):
    candidates = [
        row
        for row in rows
        if row.get("status") == "success" and row.get(metric) is not None
    ]
    if not candidates:
        return "unavailable"
    best = max(candidates, key=lambda row: row[metric])
    delta_key = "delta_avg_vs_original" if metric == "average_accuracy" else "delta_final_vs_original"
    suffix = f", delta {fmt(best.get(delta_key))}" if metric != "novel_final_accuracy" else ""
    return f"{best['variant_id']} ({fmt(best[metric])}{suffix})"


def c6_improvement_statement(rows, dataset_keyword):
    row = c6_row_for_keyword(rows, dataset_keyword)
    if not row:
        return "unavailable"
    deltas = available_deltas(row)
    if not deltas:
        return "unavailable"
    if all(delta > 0 for delta in deltas):
        return "yes"
    if any(delta > 0 for delta in deltas):
        return "mixed"
    return "no"


def c6_hurt_statement(rows, dataset_keyword):
    row = c6_row_for_keyword(rows, dataset_keyword)
    if not row:
        return "unavailable"
    deltas = available_deltas(row)
    if not deltas:
        return "unavailable"
    return "yes" if any(delta < 0 for delta in deltas) else "no"


def recommendation(rows):
    cub_statement = c6_improvement_statement(rows, "cub")
    cifar_hurts = c6_hurt_statement(rows, "cifar")
    if cub_statement == "yes" and cifar_hurts == "no":
        return "keep"
    if cub_statement in {"yes", "mixed"} and cifar_hurts == "yes":
        return "tune more"
    if cub_statement == "no" and cifar_hurts == "yes":
        return "reject"
    return "tune more"


def suspicious_drops(rows):
    messages = []
    for row in rows:
        delta_final = row.get("delta_final_vs_original")
        if delta_final is not None and delta_final < -5:
            messages.append(
                f"{row['dataset']} {row['variant_id']} final accuracy drop {abs(delta_final):.3f} points"
            )
    return messages


def row_by_variant(rows, variant_id):
    return next((row for row in rows if row["variant_id"] == variant_id), None)


def c6_row_for_keyword(rows, dataset_keyword):
    keyword = dataset_keyword.lower()
    return next(
        (
            row
            for row in rows
            if keyword in row["dataset"].lower() and row["variant_id"] == "R1_C6_ORIGINAL"
        ),
        None,
    )


def available_deltas(row):
    return [
        delta
        for delta in (row.get("delta_avg_vs_original"), row.get("delta_final_vs_original"))
        if delta is not None
    ]


def write_variant_report(output_dir, dataset, variant, result):
    metrics = normalize_metric_row(result.get("raw_metrics"))
    lines = [
        "# C6 Refinement Run",
        "",
        f"- Dataset: {dataset}",
        f"- Variant: {variant.variant_id} ({variant.name})",
        f"- Status: {result.get('status')}",
        f"- Duration sec: {fmt(result.get('duration_sec'))}",
        "",
        "## Metrics",
        "",
        f"- Average accuracy: {fmt(metrics.get('average_accuracy'))}",
        f"- Final accuracy: {fmt(metrics.get('final_accuracy'))}",
        f"- Performance degradation: {fmt(metrics.get('performance_degradation'))}",
        f"- Base final accuracy: {fmt(metrics.get('base_final_accuracy'))}",
        f"- Novel final accuracy: {fmt(metrics.get('novel_final_accuracy'))}",
        "",
        "## Purpose",
        "",
        variant.purpose,
    ]
    if variant.notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in variant.notes)
    if result.get("error_message"):
        lines.extend(["", "## Error", "", result["error_message"]])
    lines.append("")
    with open(os.path.join(output_dir, "report.md"), "w") as handle:
        handle.write("\n".join(lines))


def write_failed_variant_files(output_dir, config_preview, dataset, variant, error_message, duration_sec):
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
    write_variant_report(
        output_dir,
        dataset,
        variant,
        {
            "status": "failed",
            "duration_sec": duration_sec,
            "output_dir": output_dir,
            "raw_metrics": {},
            "error_message": error_message,
        },
    )


def build_run_manifest(timestamp, datasets, config, device, gpu_id, dry_run, seed, variants, planned_runs, git_info=None):
    return {
        "timestamp": timestamp,
        "git": git_info or {},
        "datasets": list(datasets),
        "config": config,
        "device": device,
        "gpu_id": gpu_id,
        "dry_run": dry_run,
        "seed": seed,
        "variants": [
            {
                "variant_id": variant.variant_id,
                "variant_name": variant.name,
                "purpose": variant.purpose,
                "required_phases": list(variant.required_phases),
                "disable_phases": list(variant.disable_phases),
                "notes": list(variant.notes),
            }
            for variant in variants
        ],
        "planned_runs": serialize_planned_runs(planned_runs),
    }


def serialize_planned_runs(planned_runs):
    serialized = []
    for plan in planned_runs:
        variant = plan["variant"]
        serialized.append(
            {
                "dataset": plan["dataset"],
                "variant_id": variant.variant_id,
                "variant_name": variant.name,
                "abstract_overrides": plan.get("abstract_overrides", {}),
                "resolved_overrides": plan.get("resolved_overrides", {}),
                "mapped_overrides": plan.get("mapped_overrides", {}),
                "optional_skipped": plan.get("optional_skipped", []),
                "validation_errors": plan.get("validation_errors", []),
                "status": plan.get("status", "planned"),
                "duration_sec": plan.get("duration_sec"),
                "output_dir": plan.get("output_dir"),
                "error_message": plan.get("error_message"),
            }
        )
    return serialized


def print_dry_run_plan(planned_runs):
    for plan in planned_runs:
        variant = plan["variant"]
        print(f"{plan['dataset']} / {variant.variant_id} ({variant.name})")
        printed_values = set()
        for key, value in plan["abstract_overrides"].items():
            mapped = plan.get("mapped_overrides", {}).get(key, "unresolved")
            print(f"  {key} = {value} -> {mapped}")
            printed_values.add(mapped)
        for key, value in sorted(plan.get("resolved_overrides", {}).items()):
            if key not in printed_values:
                print(f"  {key} = {value}")
        if plan["validation_errors"]:
            print("  validation errors:")
            for error in plan["validation_errors"]:
                print(f"    - {error}")


if __name__ == "__main__":
    raise SystemExit(main())
