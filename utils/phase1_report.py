import csv
import json
import os
from datetime import datetime


def cfg_to_dict(cfg):
    if isinstance(cfg, dict):
        return {key: cfg_to_dict(value) for key, value in cfg.items()}
    if hasattr(cfg, "items"):
        return {key: cfg_to_dict(value) for key, value in cfg.items()}
    if isinstance(cfg, (list, tuple)):
        return [cfg_to_dict(value) for value in cfg]
    return cfg


def write_phase1_outputs(
    cfg,
    dataset_name,
    session_metrics,
    beta_session_records,
    beta_class_records=None,
    comparison_rows=None,
    notes=None,
):
    run_dir = _phase1_run_dir(cfg, dataset_name)
    os.makedirs(run_dir, exist_ok=True)

    metrics = _build_metrics(session_metrics, beta_session_records)
    settings = _method_settings(cfg)
    run_name = _cfg_get(cfg, "RUN_NAME", "")
    summary = {
        "run_name": run_name,
        "run_dir": run_dir,
        "settings": settings,
        "metrics": metrics,
        "session_metrics": session_metrics,
        "beta_session_records": beta_session_records,
        "beta_class_records": beta_class_records or [],
    }

    _write_json(os.path.join(run_dir, "config.json"), cfg_to_dict(cfg))
    _write_json(os.path.join(run_dir, "metrics.json"), metrics)
    _write_session_accuracy_csv(os.path.join(run_dir, "session_accuracy.csv"), session_metrics)
    _write_beta_stats_csv(os.path.join(run_dir, "beta_stats.csv"), beta_session_records)
    if beta_class_records:
        _write_beta_by_class_csv(os.path.join(run_dir, "beta_by_class.csv"), beta_class_records)

    report = _render_markdown_report(
        dataset_name=dataset_name,
        cfg=cfg,
        metrics=metrics,
        session_metrics=session_metrics,
        beta_session_records=beta_session_records,
        comparison_rows=comparison_rows,
        notes=notes or [],
    )
    with open(os.path.join(run_dir, "phase1_report.md"), "w") as handle:
        handle.write(report)

    return summary


def write_phase1_ablation_summary(parent_dir, dataset_name, cfg, run_summaries):
    os.makedirs(parent_dir, exist_ok=True)
    comparison_rows = build_comparison_rows(run_summaries)

    _write_json(
        os.path.join(parent_dir, "phase1_ablation_summary.json"),
        {"dataset": dataset_name, "comparison": comparison_rows},
    )
    _write_comparison_csv(os.path.join(parent_dir, "ablation_comparison.csv"), comparison_rows)

    report = _render_ablation_markdown(dataset_name, cfg, comparison_rows)
    report_path = os.path.join(parent_dir, "phase1_report.md")
    with open(report_path, "w") as handle:
        handle.write(report)

    return {
        "run_dir": parent_dir,
        "report_path": report_path,
        "comparison_rows": comparison_rows,
    }


def build_comparison_rows(run_summaries):
    rows = []
    if not run_summaries:
        return rows

    baseline = run_summaries[0]["metrics"]
    baseline_avg = baseline["average_accuracy"]
    baseline_last = baseline["final_accuracy"]

    for summary in run_summaries:
        metrics = summary["metrics"]
        settings = summary["settings"]
        row = {
            "run_name": summary.get("run_name", ""),
            "fusion_beta_mode": settings["fusion_beta_mode"],
            "fusion_geometry": settings["fusion_geometry"],
            "reliability_mode": settings["reliability_mode"],
            "final_accuracy": metrics["final_accuracy"],
            "average_accuracy": metrics["average_accuracy"],
            "performance_degradation": metrics["performance_degradation"],
            "base_class_final_accuracy": metrics["base_class_final_accuracy"],
            "novel_class_final_accuracy": metrics["novel_class_final_accuracy"],
            "delta_average_accuracy": _round_or_none(metrics["average_accuracy"] - baseline_avg),
            "delta_final_accuracy": _round_or_none(metrics["final_accuracy"] - baseline_last),
            "is_baseline": summary is run_summaries[0],
        }
        rows.append(row)
    return rows


def _build_metrics(session_metrics, beta_session_records):
    session_acc = [float(item["mean_acc"]) for item in session_metrics]
    final_accuracy = session_acc[-1] if session_acc else None
    average_accuracy = sum(session_acc) / len(session_acc) if session_acc else None
    performance_degradation = session_acc[0] - session_acc[-1] if len(session_acc) >= 2 else 0.0
    last_session = session_metrics[-1] if session_metrics else {}

    return {
        "session_accuracy": session_acc,
        "final_accuracy": _round_or_none(final_accuracy),
        "average_accuracy": _round_or_none(average_accuracy),
        "performance_degradation": _round_or_none(performance_degradation),
        "base_class_final_accuracy": _round_or_none(last_session.get("base_avg_acc")),
        "novel_class_final_accuracy": _round_or_none(last_session.get("inc_avg_acc")),
        "beta_statistics": _aggregate_beta_stats(beta_session_records),
        "beta_by_session": beta_session_records,
    }


def _aggregate_beta_stats(beta_session_records):
    valid_records = [
        record for record in beta_session_records
        if record.get("beta_count", 0) and record.get("beta_mean") is not None
    ]
    if not valid_records:
        return {
            "beta_mean": None,
            "beta_std": None,
            "beta_min": None,
            "beta_max": None,
        }

    total = sum(record["beta_count"] for record in valid_records)
    mean = sum(record["beta_count"] * record["beta_mean"] for record in valid_records) / total
    variance = sum(
        record["beta_count"] * (record["beta_std"] ** 2 + (record["beta_mean"] - mean) ** 2)
        for record in valid_records
    ) / total
    return {
        "beta_mean": _round_or_none(mean),
        "beta_std": _round_or_none(variance ** 0.5),
        "beta_min": min(record["beta_min"] for record in valid_records),
        "beta_max": max(record["beta_max"] for record in valid_records),
    }


def _phase1_run_dir(cfg, dataset_name):
    output_dir = _cfg_get(cfg, "OUTPUT_DIR", "results/phase1")
    timestamp = _cfg_get(cfg, "PHASE1_TIMESTAMP", "") or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = _cfg_get(cfg, "RUN_NAME", "")
    path = os.path.join(output_dir, _safe_name(dataset_name), timestamp)
    if run_name:
        path = os.path.join(path, _safe_name(run_name))
    return path


def _method_settings(cfg):
    opts = cfg.TRAINER.BiMC
    return {
        "fusion_beta_mode": opts.FUSION_BETA_MODE,
        "fusion_geometry": opts.FUSION_GEOMETRY,
        "beta_temperature": opts.BETA_TEMPERATURE,
        "beta_clip_min": opts.BETA_CLIP_MIN,
        "beta_clip_max": opts.BETA_CLIP_MAX,
        "reliability_mode": opts.RELIABILITY_MODE,
        "dataset_beta": cfg.DATASET.BETA,
    }


def _render_markdown_report(
    dataset_name,
    cfg,
    metrics,
    session_metrics,
    beta_session_records,
    comparison_rows,
    notes,
):
    settings = _method_settings(cfg)
    lines = [
        "# Phase One Report",
        "",
        f"- Date/time: {datetime.now().isoformat(timespec='seconds')}",
        f"- Dataset: {dataset_name}",
        f"- Backbone: {cfg.MODEL.BACKBONE.NAME}",
        f"- Fusion beta mode: {settings['fusion_beta_mode']}",
        f"- Fusion geometry: {settings['fusion_geometry']}",
        f"- Reliability mode: {settings['reliability_mode']}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Final accuracy | {_fmt(metrics['final_accuracy'])} |",
        f"| Average accuracy | {_fmt(metrics['average_accuracy'])} |",
        f"| Performance degradation | {_fmt(metrics['performance_degradation'])} |",
        f"| Base-class final accuracy | {_fmt(metrics['base_class_final_accuracy'])} |",
        f"| Novel-class final accuracy | {_fmt(metrics['novel_class_final_accuracy'])} |",
        "",
        "## Baseline Comparison",
        "",
    ]

    if comparison_rows:
        lines.extend(_comparison_table(comparison_rows))
    else:
        lines.append("Baseline comparison is unavailable for this standalone run.")

    lines.extend([
        "",
        "## Session Accuracy",
        "",
        "| Session | Mean Acc | Base Acc | Novel Acc | Harmonic Acc | Task Acc |",
        "| ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for session_id, record in enumerate(session_metrics):
        lines.append(
            "| "
            f"{session_id} | {_fmt(record.get('mean_acc'))} | {_fmt(record.get('base_avg_acc'))} | "
            f"{_fmt(record.get('inc_avg_acc'))} | {_fmt(record.get('harmonic_acc'))} | "
            f"{json.dumps(record.get('task_acc', []))} |"
        )

    lines.extend([
        "",
        "## Beta Statistics",
        "",
    ])
    lines.extend(_beta_table(beta_session_records))
    lines.extend([
        "",
        "## Interpretation",
        "",
    ])
    lines.extend(_interpretation(settings, metrics, comparison_rows))

    if notes:
        lines.extend(["", "## Notes", ""])
        for note in notes:
            lines.append(f"- {note}")

    lines.append("")
    return "\n".join(lines)


def _render_ablation_markdown(dataset_name, cfg, comparison_rows):
    lines = [
        "# Phase One Ablation Report",
        "",
        f"- Date/time: {datetime.now().isoformat(timespec='seconds')}",
        f"- Dataset: {dataset_name}",
        f"- Backbone: {cfg.MODEL.BACKBONE.NAME}",
        "",
        "## Comparison",
        "",
    ]
    lines.extend(_comparison_table(comparison_rows))
    lines.extend(["", "## Interpretation", ""])
    lines.extend(_aggregate_interpretation(comparison_rows))
    lines.append("")
    return "\n".join(lines)


def _comparison_table(comparison_rows):
    lines = [
        "| Run | Beta Mode | Geometry | Avg | Last | Delta Avg | Delta Last | Base Final | Novel Final |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison_rows:
        lines.append(
            "| "
            f"{row['run_name']} | {row['fusion_beta_mode']} | {row['fusion_geometry']} | "
            f"{_fmt(row['average_accuracy'])} | {_fmt(row['final_accuracy'])} | "
            f"{_fmt(row['delta_average_accuracy'])} | {_fmt(row['delta_final_accuracy'])} | "
            f"{_fmt(row['base_class_final_accuracy'])} | {_fmt(row['novel_class_final_accuracy'])} |"
        )
    return lines


def _beta_table(beta_session_records):
    if not beta_session_records:
        return ["No beta statistics were recorded."]
    lines = [
        "| Session | Count | Mean | Std | Min | Max | Q05 | Q50 | Q95 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in beta_session_records:
        lines.append(
            "| "
            f"{record.get('session')} | {record.get('beta_count')} | {_fmt(record.get('beta_mean'))} | "
            f"{_fmt(record.get('beta_std'))} | {_fmt(record.get('beta_min'))} | {_fmt(record.get('beta_max'))} | "
            f"{_fmt(record.get('beta_q05'))} | {_fmt(record.get('beta_q50'))} | {_fmt(record.get('beta_q95'))} |"
        )
    return lines


def _interpretation(settings, metrics, comparison_rows):
    if not comparison_rows:
        return [
            "- Dynamic beta improved Avg: needs more testing.",
            "- Dynamic beta improved Last: needs more testing.",
            "- SLERP helped: needs paired linear and SLERP runs.",
            "- Base or novel accuracy changed: needs baseline comparison.",
            "- Recommendation: needs more testing.",
        ]

    current_name = _cfg_run_name_from_rows(comparison_rows)
    baseline = comparison_rows[0]
    current = next((row for row in comparison_rows if row["run_name"] == current_name), comparison_rows[-1])
    avg_delta = current["delta_average_accuracy"]
    last_delta = current["delta_final_accuracy"]
    rec = _recommendation(avg_delta, last_delta)
    base_delta = _delta_text(current["base_class_final_accuracy"], baseline["base_class_final_accuracy"])
    novel_delta = _delta_text(current["novel_class_final_accuracy"], baseline["novel_class_final_accuracy"])

    return [
        f"- Dynamic beta improved Avg: {_yes_no(avg_delta)} ({_fmt(avg_delta)}).",
        f"- Dynamic beta improved Last: {_yes_no(last_delta)} ({_fmt(last_delta)}).",
        f"- SLERP helped: {_slerp_helped(settings, comparison_rows)}.",
        f"- Base accuracy change: {base_delta}; novel accuracy change: {novel_delta}.",
        f"- Recommendation: {rec}.",
    ]


def _aggregate_interpretation(comparison_rows):
    if not comparison_rows:
        return ["- Recommendation: needs more testing."]
    baseline = comparison_rows[0]
    best_avg = max(comparison_rows, key=lambda row: row["average_accuracy"])
    best_last = max(comparison_rows, key=lambda row: row["final_accuracy"])
    dynamic_rows = [row for row in comparison_rows if row["fusion_beta_mode"] != "fixed"]
    best_dynamic = max(dynamic_rows, key=lambda row: row["average_accuracy"]) if dynamic_rows else None

    if best_dynamic:
        dynamic_avg = _yes_no(best_dynamic["average_accuracy"] - baseline["average_accuracy"])
        dynamic_last = _yes_no(best_dynamic["final_accuracy"] - baseline["final_accuracy"])
    else:
        dynamic_avg = "needs more testing"
        dynamic_last = "needs more testing"

    slerp_pairs = _paired_slerp_results(comparison_rows)
    if slerp_pairs:
        helped = any(pair["slerp"]["average_accuracy"] > pair["linear"]["average_accuracy"] for pair in slerp_pairs)
        slerp_text = "yes" if helped else "no"
    else:
        slerp_text = "needs more testing"

    recommendation = "keep" if best_avg is not baseline and best_last is not baseline else "needs more testing"
    return [
        f"- Dynamic beta improved Avg: {dynamic_avg}.",
        f"- Dynamic beta improved Last: {dynamic_last}.",
        f"- SLERP helped: {slerp_text}.",
        f"- Best Avg: {best_avg['run_name']} ({_fmt(best_avg['average_accuracy'])}).",
        f"- Best Last: {best_last['run_name']} ({_fmt(best_last['final_accuracy'])}).",
        f"- Recommendation: {recommendation}.",
    ]


def _paired_slerp_results(rows):
    pairs = []
    for row in rows:
        if row["fusion_geometry"] != "linear":
            continue
        match = next(
            (
                candidate for candidate in rows
                if candidate["fusion_geometry"] == "slerp"
                and candidate["fusion_beta_mode"] == row["fusion_beta_mode"]
                and candidate["reliability_mode"] == row["reliability_mode"]
            ),
            None,
        )
        if match:
            pairs.append({"linear": row, "slerp": match})
    return pairs


def _slerp_helped(settings, comparison_rows):
    target_mode = settings["fusion_beta_mode"]
    target_rel = settings["reliability_mode"]
    linear = next(
        (
            row for row in comparison_rows
            if row["fusion_beta_mode"] == target_mode
            and row["fusion_geometry"] == "linear"
            and row["reliability_mode"] == target_rel
        ),
        None,
    )
    slerp_row = next(
        (
            row for row in comparison_rows
            if row["fusion_beta_mode"] == target_mode
            and row["fusion_geometry"] == "slerp"
            and row["reliability_mode"] == target_rel
        ),
        None,
    )
    if not linear or not slerp_row:
        return "needs paired linear and SLERP runs"
    return _yes_no(slerp_row["average_accuracy"] - linear["average_accuracy"])


def _write_session_accuracy_csv(path, session_metrics):
    fieldnames = ["session", "mean_acc", "base_avg_acc", "inc_avg_acc", "harmonic_acc", "task_acc"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for session_id, record in enumerate(session_metrics):
            writer.writerow({
                "session": session_id,
                "mean_acc": record.get("mean_acc"),
                "base_avg_acc": record.get("base_avg_acc"),
                "inc_avg_acc": record.get("inc_avg_acc"),
                "harmonic_acc": record.get("harmonic_acc"),
                "task_acc": json.dumps(record.get("task_acc", [])),
            })


def _write_beta_stats_csv(path, records):
    fieldnames = [
        "session",
        "beta_count",
        "beta_mean",
        "beta_std",
        "beta_min",
        "beta_max",
        "beta_q05",
        "beta_q25",
        "beta_q50",
        "beta_q75",
        "beta_q95",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in fieldnames})


def _write_beta_by_class_csv(path, records):
    fieldnames = ["session", "class_id", "beta"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in fieldnames})


def _write_comparison_csv(path, records):
    if not records:
        return
    fieldnames = list(records[0].keys())
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


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


def _cfg_get(cfg, name, default):
    try:
        return cfg[name]
    except KeyError:
        return default


def _cfg_run_name_from_rows(rows):
    return rows[-1]["run_name"] if rows else ""


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


def _yes_no(delta):
    if delta is None:
        return "needs more testing"
    return "yes" if delta > 0 else "no"


def _delta_text(current, baseline):
    if current is None or baseline is None:
        return "n/a"
    delta = current - baseline
    return f"{_fmt(delta)}"


def _recommendation(avg_delta, last_delta):
    if avg_delta is None or last_delta is None:
        return "needs more testing"
    if avg_delta > 0 and last_delta >= 0:
        return "keep"
    if avg_delta < 0 and last_delta < 0:
        return "reject"
    return "needs more testing"
