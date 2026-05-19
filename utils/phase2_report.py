import csv
import json
import os
from datetime import datetime

from utils.phase1_report import cfg_to_dict


ABLATION_LABELS = {
    "p2_a0_original": "P2_A0 original",
    "p2_a1_robust_visual": "P2_A1 robust visual",
    "p2_a2_shrinkage_visual": "P2_A2 shrinkage visual",
    "p2_a3_dynamic_lambda": "P2_A3 dynamic lambda",
    "p2_a4_desc_discriminative": "P2_A4 description discriminative",
    "p2_a5_desc_visual_grounded": "P2_A5 description visual grounded",
    "p2_a6_desc_combined": "P2_A6 description combined",
    "p2_a7_shrinkage_desc_combined": "P2_A7 shrinkage + description combined",
    "p2_a8_shrinkage_dynamic_lambda_desc_combined": "P2_A8 shrinkage + dynamic lambda + description combined",
}


def write_phase2_outputs(
    cfg,
    dataset_name,
    session_metrics,
    phase2_records,
    comparison_rows=None,
    notes=None,
):
    run_dir = phase2_run_dir(cfg, dataset_name)
    os.makedirs(run_dir, exist_ok=True)

    run_name = _cfg_get(cfg, "RUN_NAME", "")
    metrics = build_phase2_metrics(session_metrics, phase2_records)
    settings = phase2_settings(cfg)
    summary = {
        "run_name": run_name,
        "run_dir": run_dir,
        "settings": settings,
        "metrics": metrics,
        "session_metrics": session_metrics,
        "phase2_records": phase2_records,
    }

    _write_json(os.path.join(run_dir, "config.json"), cfg_to_dict(cfg))
    _write_json(os.path.join(run_dir, "metrics.json"), metrics)
    _write_session_accuracy_csv(os.path.join(run_dir, "session_accuracy.csv"), session_metrics, run_name=None)
    _write_phase2_record_files(run_dir, phase2_records)

    report = _render_run_markdown(
        dataset_name=dataset_name,
        cfg=cfg,
        metrics=metrics,
        settings=settings,
        session_metrics=session_metrics,
        comparison_rows=comparison_rows,
        notes=notes or [],
    )
    with open(os.path.join(run_dir, "phase2_report.md"), "w") as handle:
        handle.write(report)

    return summary


def write_phase2_ablation_summary(parent_dir, dataset_name, cfg, run_summaries):
    os.makedirs(parent_dir, exist_ok=True)
    comparison_rows = build_phase2_comparison_rows(run_summaries)
    all_records = _records_with_run_names(run_summaries)

    _write_json(
        os.path.join(parent_dir, "aggregate_phase2_metrics.json"),
        {
            "dataset": dataset_name,
            "config": cfg_to_dict(cfg),
            "comparison": comparison_rows,
            "runs": [
                {
                    "run_name": summary.get("run_name", ""),
                    "settings": summary.get("settings", {}),
                    "metrics": summary.get("metrics", {}),
                }
                for summary in run_summaries
            ],
        },
    )
    _write_csv(os.path.join(parent_dir, "aggregate_phase2_metrics.csv"), comparison_rows)
    _write_combined_session_accuracy_csv(os.path.join(parent_dir, "session_accuracy.csv"), run_summaries)
    _write_phase2_record_files(parent_dir, all_records)

    report = _render_aggregate_markdown(dataset_name, cfg, comparison_rows)
    report_path = os.path.join(parent_dir, "aggregate_phase2_report.md")
    with open(report_path, "w") as handle:
        handle.write(report)

    return {
        "run_dir": parent_dir,
        "report_path": report_path,
        "comparison_rows": comparison_rows,
    }


def build_phase2_metrics(session_metrics, phase2_records):
    session_acc = [float(item["mean_acc"]) for item in session_metrics]
    final_accuracy = session_acc[-1] if session_acc else None
    average_accuracy = sum(session_acc) / len(session_acc) if session_acc else None
    performance_degradation = session_acc[0] - session_acc[-1] if len(session_acc) >= 2 else 0.0
    last_session = session_metrics[-1] if session_metrics else {}
    stats = aggregate_phase2_stats(phase2_records)

    metrics = {
        "session_accuracy": session_acc,
        "final_accuracy": _round_or_none(final_accuracy),
        "average_accuracy": _round_or_none(average_accuracy),
        "performance_degradation": _round_or_none(performance_degradation),
        "base_class_final_accuracy": _round_or_none(last_session.get("base_avg_acc")),
        "novel_class_final_accuracy": _round_or_none(last_session.get("inc_avg_acc")),
    }
    metrics.update(stats)
    return metrics


def aggregate_phase2_stats(records):
    spec = {
        "visual_quality": "visual_quality",
        "shrinkage_rho": "shrinkage_rho",
        "lambda_i": "lambda_i",
        "desc_weight_entropy": "desc_weight_entropy",
    }
    output = {}
    for metric_name, record_key in spec.items():
        values = [
            float(record[record_key])
            for record in records
            if record.get(record_key) is not None
        ]
        stats = _value_stats(values)
        for key, value in stats.items():
            output[f"{metric_name}_{key}"] = value
    return output


def build_phase2_comparison_rows(run_summaries):
    rows = []
    if not run_summaries:
        return rows

    baseline = run_summaries[0]["metrics"]
    baseline_avg = baseline["average_accuracy"]
    baseline_final = baseline["final_accuracy"]
    baseline_pd = baseline["performance_degradation"]

    for summary in run_summaries:
        metrics = summary["metrics"]
        settings = summary["settings"]
        run_name = summary.get("run_name", "")
        row = {
            "run_name": run_name,
            "label": ABLATION_LABELS.get(run_name, run_name),
            "phase2_enabled": settings["phase2_enabled"],
            "visual_proto_mode": settings["visual_proto_mode"],
            "text_proto_mode": settings["text_proto_mode"],
            "dynamic_lambda_i": settings["dynamic_lambda_i"],
            "fusion_beta_mode": settings["fusion_beta_mode"],
            "fusion_geometry": settings["fusion_geometry"],
            "final_accuracy": metrics["final_accuracy"],
            "average_accuracy": metrics["average_accuracy"],
            "performance_degradation": metrics["performance_degradation"],
            "base_class_final_accuracy": metrics["base_class_final_accuracy"],
            "novel_class_final_accuracy": metrics["novel_class_final_accuracy"],
            "delta_avg": _round_or_none(metrics["average_accuracy"] - baseline_avg),
            "delta_final": _round_or_none(metrics["final_accuracy"] - baseline_final),
            "delta_pd": _round_or_none(baseline_pd - metrics["performance_degradation"]),
            "visual_quality_mean": metrics["visual_quality_mean"],
            "visual_quality_std": metrics["visual_quality_std"],
            "visual_quality_min": metrics["visual_quality_min"],
            "visual_quality_max": metrics["visual_quality_max"],
            "shrinkage_rho_mean": metrics["shrinkage_rho_mean"],
            "shrinkage_rho_std": metrics["shrinkage_rho_std"],
            "shrinkage_rho_min": metrics["shrinkage_rho_min"],
            "shrinkage_rho_max": metrics["shrinkage_rho_max"],
            "lambda_i_mean": metrics["lambda_i_mean"],
            "lambda_i_std": metrics["lambda_i_std"],
            "lambda_i_min": metrics["lambda_i_min"],
            "lambda_i_max": metrics["lambda_i_max"],
            "desc_weight_entropy_mean": metrics["desc_weight_entropy_mean"],
            "desc_weight_entropy_std": metrics["desc_weight_entropy_std"],
            "desc_weight_entropy_min": metrics["desc_weight_entropy_min"],
            "desc_weight_entropy_max": metrics["desc_weight_entropy_max"],
            "is_baseline": summary is run_summaries[0],
        }
        rows.append(row)
    return rows


def phase2_settings(cfg):
    opts = cfg.TRAINER.BiMC
    p2 = opts.PHASE2
    return {
        "phase2_enabled": bool(p2.ENABLED),
        "visual_proto_mode": p2.VISUAL_PROTO_MODE,
        "robust_kappa": p2.ROBUST_KAPPA,
        "robust_drop_lowest": bool(p2.ROBUST_DROP_LOWEST),
        "shrinkage_prior": p2.SHRINKAGE_PRIOR,
        "shrinkage_min": p2.SHRINKAGE_MIN,
        "shrinkage_max": p2.SHRINKAGE_MAX,
        "dynamic_lambda_i": bool(p2.DYNAMIC_LAMBDA_I),
        "dynamic_lambda_min": p2.DYNAMIC_LAMBDA_MIN,
        "dynamic_lambda_max": p2.DYNAMIC_LAMBDA_MAX,
        "text_proto_mode": p2.TEXT_PROTO_MODE,
        "desc_temp": p2.DESC_TEMP,
        "desc_lambda_d": p2.DESC_LAMBDA_D,
        "desc_topk": p2.DESC_TOPK,
        "fusion_beta_mode": opts.FUSION_BETA_MODE,
        "fusion_geometry": opts.FUSION_GEOMETRY,
        "dataset_beta": cfg.DATASET.BETA,
    }


def phase2_run_dir(cfg, dataset_name):
    output_dir = _cfg_get(cfg, "OUTPUT_DIR", "results/phase2")
    timestamp = _cfg_get(cfg, "PHASE2_TIMESTAMP", "") or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = _cfg_get(cfg, "RUN_NAME", "")
    path = os.path.join(output_dir, _safe_name(dataset_name), timestamp)
    if run_name:
        path = os.path.join(path, _safe_name(run_name))
    return path


def _render_run_markdown(dataset_name, cfg, metrics, settings, session_metrics, comparison_rows, notes):
    lines = [
        "# Phase Two Report",
        "",
        f"- Date/time: {datetime.now().isoformat(timespec='seconds')}",
        f"- Dataset: {dataset_name}",
        f"- Backbone: {cfg.MODEL.BACKBONE.NAME}",
        f"- Run: {_cfg_get(cfg, 'RUN_NAME', '')}",
        "",
        "## Config Used",
        "",
        "| Setting | Value |",
        "| --- | --- |",
    ]
    for key in [
        "phase2_enabled",
        "visual_proto_mode",
        "text_proto_mode",
        "dynamic_lambda_i",
        "fusion_beta_mode",
        "fusion_geometry",
        "shrinkage_prior",
        "desc_temp",
        "desc_lambda_d",
        "dataset_beta",
    ]:
        lines.append(f"| {key} | {settings.get(key)} |")

    lines.extend([
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Average accuracy | {_fmt(metrics['average_accuracy'])} |",
        f"| Final accuracy | {_fmt(metrics['final_accuracy'])} |",
        f"| Performance degradation | {_fmt(metrics['performance_degradation'])} |",
        f"| Base final accuracy | {_fmt(metrics['base_class_final_accuracy'])} |",
        f"| Novel final accuracy | {_fmt(metrics['novel_class_final_accuracy'])} |",
        "",
        "## Session Accuracy",
        "",
    ])
    lines.extend(_session_table(session_metrics))

    lines.extend(["", "## Prototype Statistics", ""])
    lines.extend(_prototype_stats_table(metrics))

    lines.extend(["", "## Baseline Comparison", ""])
    if comparison_rows:
        lines.extend(_comparison_table(comparison_rows))
    else:
        lines.append("Baseline comparison is unavailable for this standalone run.")

    if notes:
        lines.extend(["", "## Notes", ""])
        for note in notes:
            lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _render_aggregate_markdown(dataset_name, cfg, comparison_rows):
    lines = [
        "# Phase Two Ablation Report",
        "",
        f"- Date/time: {datetime.now().isoformat(timespec='seconds')}",
        f"- Dataset: {dataset_name}",
        f"- Backbone: {cfg.MODEL.BACKBONE.NAME}",
        "",
        "## Config Used",
        "",
        "| Setting | Value |",
        "| --- | --- |",
        f"| Seed | {cfg.SEED} |",
        f"| Trainer | BiMC |",
        f"| Fusion beta mode | fixed |",
        f"| Fusion geometry | linear |",
        f"| Dataset beta | {cfg.DATASET.BETA} |",
        f"| Text calibration | {cfg.TRAINER.BiMC.TEXT_CALIBRATION} |",
        f"| Vision calibration | {cfg.TRAINER.BiMC.VISION_CALIBRATION} |",
        "",
        "## Comparison",
        "",
    ]
    lines.extend(_comparison_table(comparison_rows))
    lines.extend(["", "## Interpretation", ""])
    lines.extend(_aggregate_interpretation(comparison_rows))
    lines.append("")
    return "\n".join(lines)


def _comparison_table(rows):
    if not rows:
        return ["No runs were recorded."]
    lines = [
        "| Run | Visual | Text | Dyn Lambda | Avg | Final | PD | Delta Avg | Delta Final | Delta PD | Base Final | Novel Final |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['run_name']} | {row['visual_proto_mode']} | {row['text_proto_mode']} | "
            f"{row['dynamic_lambda_i']} | {_fmt(row['average_accuracy'])} | {_fmt(row['final_accuracy'])} | "
            f"{_fmt(row['performance_degradation'])} | {_fmt(row['delta_avg'])} | "
            f"{_fmt(row['delta_final'])} | {_fmt(row['delta_pd'])} | "
            f"{_fmt(row['base_class_final_accuracy'])} | {_fmt(row['novel_class_final_accuracy'])} |"
        )
    return lines


def _session_table(session_metrics):
    lines = [
        "| Session | Mean Acc | Base Acc | Novel Acc | Harmonic Acc | Task Acc |",
        "| ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for session_id, record in enumerate(session_metrics):
        lines.append(
            "| "
            f"{session_id} | {_fmt(record.get('mean_acc'))} | {_fmt(record.get('base_avg_acc'))} | "
            f"{_fmt(record.get('inc_avg_acc'))} | {_fmt(record.get('harmonic_acc'))} | "
            f"{json.dumps(record.get('task_acc', []))} |"
        )
    return lines


def _prototype_stats_table(metrics):
    lines = [
        "| Statistic | Mean | Std | Min | Max |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key in ["visual_quality", "shrinkage_rho", "lambda_i", "desc_weight_entropy"]:
        lines.append(
            "| "
            f"{key} | {_fmt(metrics.get(f'{key}_mean'))} | {_fmt(metrics.get(f'{key}_std'))} | "
            f"{_fmt(metrics.get(f'{key}_min'))} | {_fmt(metrics.get(f'{key}_max'))} |"
        )
    return lines


def _aggregate_interpretation(rows):
    if not rows:
        return ["- Recommendation: needs more tests."]

    baseline = rows[0]
    best_avg = max(rows, key=lambda row: _metric_value(row["average_accuracy"]))
    best_final = max(rows, key=lambda row: _metric_value(row["final_accuracy"]))
    return [
        f"- Best method by average accuracy: {best_avg['run_name']} ({_fmt(best_avg['average_accuracy'])}).",
        f"- Best method by final accuracy: {best_final['run_name']} ({_fmt(best_final['final_accuracy'])}).",
        f"- Robust visual prototype helped: {_helped(rows, baseline, lambda row: row['visual_proto_mode'] == 'robust_weighted')}.",
        f"- Shrinkage helped: {_helped(rows, baseline, lambda row: row['visual_proto_mode'] == 'shrinkage_base_prior')}.",
        f"- Dynamic lambda helped: {_helped(rows, baseline, lambda row: row['dynamic_lambda_i'])}.",
        f"- Description reweighting helped: {_helped(rows, baseline, lambda row: row['text_proto_mode'] != 'mean')}.",
        f"- Recommendation: {_recommendation(rows, baseline)}.",
    ]


def _helped(rows, baseline, predicate):
    candidates = [row for row in rows if predicate(row)]
    if not candidates:
        return "needs more tests"
    best = max(candidates, key=lambda row: _metric_value(row["average_accuracy"]))
    avg_delta = best["average_accuracy"] - baseline["average_accuracy"]
    final_delta = best["final_accuracy"] - baseline["final_accuracy"]
    if avg_delta > 0 and final_delta >= 0:
        return f"yes ({best['run_name']}, delta_avg={_fmt(avg_delta)}, delta_final={_fmt(final_delta)})"
    return f"no ({best['run_name']}, delta_avg={_fmt(avg_delta)}, delta_final={_fmt(final_delta)})"


def _recommendation(rows, baseline):
    if len(rows) <= 1:
        return "needs more tests"
    best = max(rows, key=lambda row: _metric_value(row["average_accuracy"]))
    if best is baseline:
        return "reject" if all(row["delta_avg"] <= 0 and row["delta_final"] <= 0 for row in rows[1:]) else "needs more tests"
    if best["delta_avg"] > 0 and best["delta_final"] >= 0:
        return "keep"
    return "needs more tests"


def _write_phase2_record_files(directory, records):
    _write_csv(os.path.join(directory, "phase2_stats.csv"), records)
    _write_csv(
        os.path.join(directory, "visual_quality_by_class.csv"),
        _select_record_columns(records, ["run_name", "session", "class_id", "visual_proto_mode", "visual_quality", "visual_fallback", "num_support"]),
    )
    _write_csv(
        os.path.join(directory, "shrinkage_rho_by_class.csv"),
        _select_record_columns(records, ["run_name", "session", "class_id", "visual_proto_mode", "shrinkage_rho", "visual_fallback"]),
    )
    _write_csv(
        os.path.join(directory, "dynamic_lambda_by_class.csv"),
        _select_record_columns(records, ["run_name", "session", "class_id", "lambda_i", "lambda_source"]),
    )
    _write_csv(
        os.path.join(directory, "description_stats_by_class.csv"),
        _select_record_columns(
            records,
            [
                "run_name",
                "session",
                "class_id",
                "text_proto_mode",
                "num_descriptions",
                "desc_weight_entropy",
                "desc_weight_min",
                "desc_weight_max",
                "desc_score_mean",
                "desc_score_std",
                "desc_score_min",
                "desc_score_max",
                "text_fallback",
            ],
        ),
    )


def _select_record_columns(records, columns):
    return [{key: record.get(key) for key in columns} for record in records]


def _write_session_accuracy_csv(path, session_metrics, run_name=None):
    rows = []
    for session_id, record in enumerate(session_metrics):
        row = {
            "session": session_id,
            "mean_acc": record.get("mean_acc"),
            "base_avg_acc": record.get("base_avg_acc"),
            "inc_avg_acc": record.get("inc_avg_acc"),
            "harmonic_acc": record.get("harmonic_acc"),
            "task_acc": json.dumps(record.get("task_acc", [])),
        }
        if run_name is not None:
            row["run_name"] = run_name
        rows.append(row)
    _write_csv(path, rows)


def _write_combined_session_accuracy_csv(path, run_summaries):
    rows = []
    for summary in run_summaries:
        run_name = summary.get("run_name", "")
        for session_id, record in enumerate(summary.get("session_metrics", [])):
            rows.append({
                "run_name": run_name,
                "session": session_id,
                "mean_acc": record.get("mean_acc"),
                "base_avg_acc": record.get("base_avg_acc"),
                "inc_avg_acc": record.get("inc_avg_acc"),
                "harmonic_acc": record.get("harmonic_acc"),
                "task_acc": json.dumps(record.get("task_acc", [])),
            })
    _write_csv(path, rows)


def _records_with_run_names(run_summaries):
    records = []
    for summary in run_summaries:
        run_name = summary.get("run_name", "")
        for record in summary.get("phase2_records", []):
            row = dict(record)
            row["run_name"] = run_name
            records.append(row)
    return records


def _write_csv(path, rows):
    if not rows:
        with open(path, "w", newline="") as handle:
            handle.write("")
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _write_json(path, value):
    with open(path, "w") as handle:
        json.dump(_json_safe(value), handle, indent=2)


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def _csv_value(value):
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value)
    return value


def _value_stats(values):
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "mean": _round_or_none(mean),
        "std": _round_or_none(variance ** 0.5),
        "min": _round_or_none(min(values)),
        "max": _round_or_none(max(values)),
    }


def _metric_value(value):
    return float("-inf") if value is None else float(value)


def _cfg_get(cfg, name, default):
    try:
        return cfg[name]
    except KeyError:
        return default


def _safe_name(value):
    return str(value).lower().replace("/", "_").replace(" ", "_")


def _fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    return f"{float(value):.3f}"


def _round_or_none(value, digits=4):
    if value is None:
        return None
    return round(float(value), digits)
