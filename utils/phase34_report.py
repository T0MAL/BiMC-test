import csv
import json
import os
from datetime import datetime


def write_phase34_run_outputs(run_dir, dataset_name, cfg, summary, ablation):
    os.makedirs(run_dir, exist_ok=True)
    metrics = build_metrics(summary.get("session_metrics", []))
    run_record = {
        "ablation_id": ablation["id"],
        "run_name": ablation["run_name"],
        "status": summary.get("status", "ok"),
        "error": summary.get("error"),
        "settings": ablation.get("settings", {}),
        "metrics": metrics,
        "summary": summary,
        "run_dir": run_dir,
    }

    _write_json(os.path.join(run_dir, "config_used.json"), cfg_to_dict(cfg))
    _write_json(os.path.join(run_dir, "phase34_metrics.json"), run_record)
    _write_session_accuracy_csv(
        os.path.join(run_dir, "session_accuracy.csv"),
        summary.get("session_metrics", []),
        extra={"ablation_id": ablation["id"], "run_name": ablation["run_name"]},
    )
    _write_records_csv(os.path.join(run_dir, "phase3_score_stats.csv"), summary.get("phase3_score_records", []))
    _write_records_csv(os.path.join(run_dir, "phase4_cov_stats.csv"), summary.get("phase4_cov_records", []))
    _write_records_csv(os.path.join(run_dir, "hubness_by_class.csv"), summary.get("hubness_class_records", []))
    _write_records_csv(os.path.join(run_dir, "dynamic_alpha_stats.csv"), summary.get("dynamic_alpha_records", []))
    _write_records_csv(os.path.join(run_dir, "covariance_stats_by_class.csv"), summary.get("covariance_class_records", []))

    report = _render_run_report(dataset_name, cfg, run_record)
    with open(os.path.join(run_dir, "phase34_report.md"), "w") as handle:
        handle.write(report)
    return run_record


def write_phase34_aggregate(parent_dir, dataset_name, cfg, run_records):
    os.makedirs(parent_dir, exist_ok=True)
    rows = build_comparison_rows(run_records)

    _write_json(
        os.path.join(parent_dir, "aggregate_phase34_metrics.json"),
        {"dataset": dataset_name, "comparison": rows, "runs": run_records},
    )
    _write_records_csv(os.path.join(parent_dir, "aggregate_phase34_metrics.csv"), rows)

    session_rows = []
    phase3_rows = []
    phase4_rows = []
    hubness_rows = []
    alpha_rows = []
    covariance_rows = []
    for record in run_records:
        prefix = {"ablation_id": record["ablation_id"], "run_name": record["run_name"]}
        summary = record.get("summary", {})
        for session_id, session in enumerate(summary.get("session_metrics", [])):
            row = dict(prefix)
            row["session"] = session_id
            row.update(session)
            session_rows.append(row)
        phase3_rows.extend(_with_prefix(summary.get("phase3_score_records", []), prefix))
        phase4_rows.extend(_with_prefix(summary.get("phase4_cov_records", []), prefix))
        hubness_rows.extend(_with_prefix(summary.get("hubness_class_records", []), prefix))
        alpha_rows.extend(_with_prefix(summary.get("dynamic_alpha_records", []), prefix))
        covariance_rows.extend(_with_prefix(summary.get("covariance_class_records", []), prefix))

    _write_records_csv(os.path.join(parent_dir, "session_accuracy.csv"), session_rows)
    _write_records_csv(os.path.join(parent_dir, "phase3_score_stats.csv"), phase3_rows)
    _write_records_csv(os.path.join(parent_dir, "phase4_cov_stats.csv"), phase4_rows)
    _write_records_csv(os.path.join(parent_dir, "hubness_by_class.csv"), hubness_rows)
    _write_records_csv(os.path.join(parent_dir, "dynamic_alpha_stats.csv"), alpha_rows)
    _write_records_csv(os.path.join(parent_dir, "covariance_stats_by_class.csv"), covariance_rows)

    report = _render_aggregate_report(dataset_name, cfg, rows, run_records)
    report_path = os.path.join(parent_dir, "aggregate_phase34_report.md")
    with open(report_path, "w") as handle:
        handle.write(report)

    return {
        "run_dir": parent_dir,
        "report_path": report_path,
        "comparison_rows": rows,
    }


def build_metrics(session_metrics):
    session_acc = [float(item["mean_acc"]) for item in session_metrics if item.get("mean_acc") is not None]
    final_accuracy = session_acc[-1] if session_acc else None
    average_accuracy = sum(session_acc) / len(session_acc) if session_acc else None
    performance_degradation = session_acc[0] - session_acc[-1] if len(session_acc) >= 2 else 0.0 if session_acc else None
    last_session = session_metrics[-1] if session_metrics else {}
    return {
        "session_accuracy": session_acc,
        "average_accuracy": _round_or_none(average_accuracy),
        "final_accuracy": _round_or_none(final_accuracy),
        "performance_degradation": _round_or_none(performance_degradation),
        "base_class_final_accuracy": _round_or_none(last_session.get("base_avg_acc")),
        "novel_class_final_accuracy": _round_or_none(last_session.get("inc_avg_acc")),
    }


def build_comparison_rows(run_records):
    if not run_records:
        return []
    baseline = next((record for record in run_records if record["ablation_id"] == "P34_A0"), run_records[0])
    baseline_metrics = baseline["metrics"]
    rows = []
    for record in run_records:
        metrics = record["metrics"]
        row = {
            "ablation_id": record["ablation_id"],
            "run_name": record["run_name"],
            "status": record.get("status", "ok"),
            "average_accuracy": metrics.get("average_accuracy"),
            "final_accuracy": metrics.get("final_accuracy"),
            "performance_degradation": metrics.get("performance_degradation"),
            "base_class_final_accuracy": metrics.get("base_class_final_accuracy"),
            "novel_class_final_accuracy": metrics.get("novel_class_final_accuracy"),
            "delta_avg": _delta(metrics.get("average_accuracy"), baseline_metrics.get("average_accuracy")),
            "delta_final": _delta(metrics.get("final_accuracy"), baseline_metrics.get("final_accuracy")),
            "delta_pd": _delta(metrics.get("performance_degradation"), baseline_metrics.get("performance_degradation")),
            "error": record.get("error"),
        }
        rows.append(row)
    return rows


def markdown_table(rows):
    lines = [
        "| Ablation | Status | Avg | Final | PD | Delta Avg | Delta Final | Delta PD | Base Final | Novel Final |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['run_name']} | {row['status']} | {_fmt(row.get('average_accuracy'))} | "
            f"{_fmt(row.get('final_accuracy'))} | {_fmt(row.get('performance_degradation'))} | "
            f"{_fmt(row.get('delta_avg'))} | {_fmt(row.get('delta_final'))} | {_fmt(row.get('delta_pd'))} | "
            f"{_fmt(row.get('base_class_final_accuracy'))} | {_fmt(row.get('novel_class_final_accuracy'))} |"
        )
    return "\n".join(lines)


def cfg_to_dict(cfg):
    if isinstance(cfg, dict):
        return {key: cfg_to_dict(value) for key, value in cfg.items()}
    if hasattr(cfg, "items"):
        return {key: cfg_to_dict(value) for key, value in cfg.items()}
    if isinstance(cfg, (list, tuple)):
        return [cfg_to_dict(value) for value in cfg]
    return cfg


def _render_run_report(dataset_name, cfg, run_record):
    metrics = run_record["metrics"]
    lines = [
        "# Phase 3/4 Run Report",
        "",
        f"- Date/time: {datetime.now().isoformat(timespec='seconds')}",
        f"- Dataset: {dataset_name}",
        f"- Ablation: {run_record['run_name']}",
        f"- Status: {run_record['status']}",
        f"- Backbone: {cfg.MODEL.BACKBONE.NAME}",
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
        "## Config Used",
        "",
        f"- Phase 3 enabled: {cfg.TRAINER.BiMC.PHASE3.ENABLED}",
        f"- Phase 4 enabled: {cfg.TRAINER.BiMC.PHASE4.ENABLED}",
        f"- Phase 4 mode: {cfg.TRAINER.BiMC.PHASE4.COV_MODE}",
        "",
    ]
    if run_record.get("error"):
        lines.extend(["## Error", "", str(run_record["error"]), ""])
    return "\n".join(lines)


def _render_aggregate_report(dataset_name, cfg, rows, run_records):
    lines = [
        "# Phase 3/4 Ablation Report",
        "",
        f"- Date/time: {datetime.now().isoformat(timespec='seconds')}",
        f"- Dataset: {dataset_name}",
        f"- Backbone: {cfg.MODEL.BACKBONE.NAME}",
        "",
        "## Comparison",
        "",
        markdown_table(rows),
        "",
        "## Phase 3 Stats",
        "",
    ]
    lines.extend(_phase3_summary(run_records))
    lines.extend(["", "## Phase 4 Stats", ""])
    lines.extend(_phase4_summary(run_records))
    lines.extend(["", "## Interpretation", ""])
    lines.extend(_interpretation(rows))
    lines.append("")
    return "\n".join(lines)


def _phase3_summary(run_records):
    rows = []
    for record in run_records:
        stats = _average_records(record.get("summary", {}).get("phase3_score_records", []))
        alpha_stats = _average_records(record.get("summary", {}).get("dynamic_alpha_records", []))
        if not stats and not alpha_stats:
            continue
        rows.append(
            f"- {record['run_name']}: hubness std {_fmt(stats.get('hubness_std'))}, "
            f"alpha mean {_fmt(alpha_stats.get('alpha_mean'))}, "
            f"alpha std {_fmt(alpha_stats.get('alpha_std'))}, "
            f"energy calib {_fmt(stats.get('energy_calib_mean'))}, "
            f"energy cov {_fmt(stats.get('energy_cov_mean'))}, "
            f"energy nn {_fmt(stats.get('energy_nn_mean'))}."
        )
    return rows or ["No Phase 3 statistics were recorded."]


def _phase4_summary(run_records):
    rows = []
    for record in run_records:
        stats = _average_records(record.get("summary", {}).get("phase4_cov_records", []))
        if not stats:
            continue
        rows.append(
            f"- {record['run_name']}: variance mean {_fmt(stats.get('phase4_var_mean'))}, "
            f"variance std {_fmt(stats.get('phase4_var_std'))}, "
            f"borrowed entropy {_fmt(stats.get('borrowed_weight_entropy'))}, "
            f"novel cov score mean {_fmt(stats.get('novel_cov_score_mean'))}."
        )
    return rows or ["No Phase 4 statistics were recorded."]


def _interpretation(rows):
    valid_rows = [row for row in rows if row.get("status") == "ok" and row.get("average_accuracy") is not None]
    if not valid_rows:
        return ["- Recommendation: needs more test."]
    baseline = next((row for row in valid_rows if row["ablation_id"] == "P34_A0"), valid_rows[0])
    best_avg = max(valid_rows, key=lambda row: row["average_accuracy"])
    best_final = max(valid_rows, key=lambda row: row["final_accuracy"])
    recommendation = "keep" if best_avg is not baseline and (best_avg.get("delta_avg") or 0) > 0 else "needs more test"
    if best_avg is baseline:
        recommendation = "reject"

    return [
        f"- Best method by average accuracy: {best_avg['run_name']} ({_fmt(best_avg['average_accuracy'])}).",
        f"- Best method by final accuracy: {best_final['run_name']} ({_fmt(best_final['final_accuracy'])}).",
        f"- Hubness helped: {_helped(rows, ['P34_A1', 'P34_A5', 'P34_A9'])}.",
        f"- Temperature scaling helped: {_helped(rows, ['P34_A2'])}.",
        f"- Dynamic alpha helped: {_helped(rows, ['P34_A3', 'P34_A5', 'P34_A9'])}.",
        f"- Energy helped: {_helped(rows, ['P34_A4'])}.",
        f"- Covariance shrinkage helped: {_helped(rows, ['P34_A6', 'P34_A8', 'P34_A9'])}.",
        f"- Base-borrowed covariance helped: {_helped(rows, ['P34_A7', 'P34_A8', 'P34_A9'])}.",
        f"- Final recommendation: {recommendation}.",
    ]


def _helped(rows, ablation_ids):
    candidates = [
        row for row in rows
        if row.get("ablation_id") in ablation_ids
        and row.get("status") == "ok"
        and row.get("delta_avg") is not None
    ]
    if not candidates:
        return "needs more test"
    best = max(candidates, key=lambda row: row["delta_avg"])
    return "yes" if best["delta_avg"] > 0 else "no"


def _average_records(records):
    values = {}
    counts = {}
    for record in records:
        for key, value in record.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and value is not None:
                values[key] = values.get(key, 0.0) + float(value)
                counts[key] = counts.get(key, 0) + 1
    return {key: values[key] / counts[key] for key in values}


def _write_session_accuracy_csv(path, session_metrics, extra=None):
    extra = extra or {}
    records = []
    for session_id, record in enumerate(session_metrics):
        row = dict(extra)
        row.update({
            "session": session_id,
            "mean_acc": record.get("mean_acc"),
            "base_avg_acc": record.get("base_avg_acc"),
            "inc_avg_acc": record.get("inc_avg_acc"),
            "harmonic_acc": record.get("harmonic_acc"),
            "task_acc": json.dumps(record.get("task_acc", [])),
        })
        records.append(row)
    _write_records_csv(path, records)


def _write_records_csv(path, records):
    records = records or []
    fieldnames = _fieldnames(records)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: _csv_value(record.get(key)) for key in fieldnames})


def _fieldnames(records):
    preferred = [
        "ablation_id",
        "run_name",
        "status",
        "session",
        "batch",
        "class_id",
        "mean_acc",
        "average_accuracy",
        "final_accuracy",
        "performance_degradation",
        "delta_avg",
        "delta_final",
        "delta_pd",
    ]
    keys = []
    for key in preferred:
        if any(key in record for record in records):
            keys.append(key)
    for record in records:
        for key in record:
            if key not in keys:
                keys.append(key)
    return keys or ["empty"]


def _write_json(path, value):
    with open(path, "w") as handle:
        json.dump(_json_safe(value), handle, indent=2)


def _with_prefix(records, prefix):
    rows = []
    for record in records:
        row = dict(prefix)
        row.update(record)
        rows.append(row)
    return rows


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
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_safe(value))
    return value


def _delta(current, baseline):
    if current is None or baseline is None:
        return None
    return _round_or_none(float(current) - float(baseline))


def _round_or_none(value, digits=4):
    if value is None:
        return None
    return round(float(value), digits)


def _fmt(value):
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def safe_name(value):
    return str(value).lower().replace("/", "_").replace(" ", "_")
