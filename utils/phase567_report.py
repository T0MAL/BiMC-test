import csv
import json
import os
from datetime import datetime

from utils.phase1_report import cfg_to_dict


ABLATION_LABELS = {
    "p567_a0_original": "P567_A0 original",
    "p567_a1_common_direction": "P567_A1 common direction",
    "p567_a2_whitening_diag": "P567_A2 diagonal whitening",
    "p567_a3_lda_shrinkage": "P567_A3 LDA shrinkage",
    "p567_a4_ot_text_image": "P567_A4 OT text-image",
    "p567_a5_support_aware_text": "P567_A5 support-aware text",
    "p567_a6_label_prior": "P567_A6 label prior",
    "p567_a7_prototype_repulsion": "P567_A7 prototype repulsion",
    "p567_a8_graph_highpass": "P567_A8 graph high-pass",
    "p567_a9_cdr_support_text": "P567_A9 CDR + support-aware text",
    "p567_a10_whitening_graph": "P567_A10 whitening + graph",
    "p567_a11_ot_label_prior": "P567_A11 OT + label prior",
    "p567_a12_best_combo_safe": "P567_A12 safe combo",
}


def write_phase567_outputs(
    cfg,
    dataset_name,
    session_metrics,
    phase5_records,
    phase6_records,
    phase7_records,
    label_prior_records,
    status="success",
    error=None,
    comparison_rows=None,
    notes=None,
):
    run_dir = phase567_run_dir(cfg, dataset_name)
    os.makedirs(run_dir, exist_ok=True)

    run_name = _cfg_get(cfg, "RUN_NAME", "")
    settings = phase567_settings(cfg)
    metrics = build_phase567_metrics(session_metrics, phase5_records, phase6_records, phase7_records, label_prior_records)
    summary = {
        "run_name": run_name,
        "run_dir": run_dir,
        "settings": settings,
        "metrics": metrics,
        "session_metrics": session_metrics,
        "phase5_records": phase5_records,
        "phase6_records": phase6_records,
        "phase7_records": phase7_records,
        "label_prior_records": label_prior_records,
        "status": status,
        "error": error,
    }

    _write_json(os.path.join(run_dir, "config.json"), cfg_to_dict(cfg))
    _write_json(os.path.join(run_dir, "metrics.json"), metrics)
    _write_session_accuracy_csv(os.path.join(run_dir, "session_accuracy.csv"), session_metrics)
    _write_csv(os.path.join(run_dir, "phase5_space_stats.csv"), phase5_records)
    _write_csv(os.path.join(run_dir, "phase6_alignment_stats.csv"), phase6_records)
    _write_csv(os.path.join(run_dir, "phase7_separation_stats.csv"), phase7_records)
    _write_csv(os.path.join(run_dir, "label_prior_stats.csv"), label_prior_records)
    _write_json(os.path.join(run_dir, "run_manifest.json"), _run_manifest(cfg, dataset_name, summary))

    report = _render_run_markdown(
        dataset_name=dataset_name,
        cfg=cfg,
        summary=summary,
        comparison_rows=comparison_rows,
        notes=notes or [],
    )
    with open(os.path.join(run_dir, "phase567_report.md"), "w") as handle:
        handle.write(report)
    return summary


def failed_phase567_summary(cfg, dataset_name, run_name, error, status="failed"):
    cfg.defrost()
    cfg.RUN_NAME = run_name
    cfg.freeze()
    run_dir = phase567_run_dir(cfg, dataset_name)
    os.makedirs(run_dir, exist_ok=True)
    summary = {
        "run_name": run_name,
        "run_dir": run_dir,
        "settings": phase567_settings(cfg),
        "metrics": build_phase567_metrics([], [], [], [], []),
        "session_metrics": [],
        "phase5_records": [],
        "phase6_records": [],
        "phase7_records": [],
        "label_prior_records": [],
        "status": status,
        "error": str(error),
    }
    _write_json(os.path.join(run_dir, "run_manifest.json"), _run_manifest(cfg, dataset_name, summary))
    with open(os.path.join(run_dir, "phase567_report.md"), "w") as handle:
        handle.write(_render_run_markdown(dataset_name, cfg, summary, comparison_rows=None, notes=[]))
    return summary


def write_phase567_ablation_summary(parent_dir, dataset_name, cfg, run_summaries, manifest=None):
    os.makedirs(parent_dir, exist_ok=True)
    rows = build_phase567_comparison_rows(run_summaries)
    run_manifest = manifest or {}
    run_manifest.update({
        "dataset": dataset_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "num_runs": len(run_summaries),
        "runs": [
            {
                "run_name": summary.get("run_name", ""),
                "status": summary.get("status", "unknown"),
                "run_dir": summary.get("run_dir"),
                "error": summary.get("error"),
            }
            for summary in run_summaries
        ],
    })

    _write_json(
        os.path.join(parent_dir, "aggregate_phase567_metrics.json"),
        {
            "dataset": dataset_name,
            "config": cfg_to_dict(cfg),
            "comparison": rows,
            "runs": _json_safe(run_summaries),
        },
    )
    _write_csv(os.path.join(parent_dir, "aggregate_phase567_metrics.csv"), rows)
    _write_combined_session_accuracy_csv(os.path.join(parent_dir, "session_accuracy.csv"), run_summaries)
    _write_csv(os.path.join(parent_dir, "phase5_space_stats.csv"), _records_with_run_names(run_summaries, "phase5_records"))
    _write_csv(os.path.join(parent_dir, "phase6_alignment_stats.csv"), _records_with_run_names(run_summaries, "phase6_records"))
    _write_csv(os.path.join(parent_dir, "phase7_separation_stats.csv"), _records_with_run_names(run_summaries, "phase7_records"))
    _write_csv(os.path.join(parent_dir, "label_prior_stats.csv"), _records_with_run_names(run_summaries, "label_prior_records"))
    _write_json(os.path.join(parent_dir, "run_manifest.json"), run_manifest)

    report = _render_aggregate_markdown(dataset_name, cfg, rows)
    report_path = os.path.join(parent_dir, "aggregate_phase567_report.md")
    with open(report_path, "w") as handle:
        handle.write(report)

    return {
        "run_dir": parent_dir,
        "report_path": report_path,
        "comparison_rows": rows,
        "manifest": run_manifest,
    }


def build_phase567_metrics(session_metrics, phase5_records, phase6_records, phase7_records, label_prior_records):
    session_acc = [float(item["mean_acc"]) for item in session_metrics]
    final_accuracy = session_acc[-1] if session_acc else None
    average_accuracy = sum(session_acc) / len(session_acc) if session_acc else None
    performance_degradation = session_acc[0] - session_acc[-1] if len(session_acc) >= 2 else 0.0
    last_session = session_metrics[-1] if session_metrics else {}
    metrics = {
        "session_accuracy": session_acc,
        "average_accuracy": _round_or_none(average_accuracy),
        "final_accuracy": _round_or_none(final_accuracy),
        "performance_degradation": _round_or_none(performance_degradation),
        "base_class_final_accuracy": _round_or_none(last_session.get("base_avg_acc")),
        "novel_class_final_accuracy": _round_or_none(last_session.get("inc_avg_acc")),
    }
    metrics.update(_aggregate_record_stats("phase5", phase5_records, ["common_direction_norm", "whitening_var_mean", "lda_dim_used", "lda_eigen_mean"]))
    metrics.update(_aggregate_record_stats("phase6", phase6_records, ["ot_cost_mean", "weight_entropy", "alignment_score_mean"]))
    metrics.update(_aggregate_record_stats("phase7", phase7_records, ["pairwise_similarity_mean", "affected_prototypes", "displacement_norm_mean"]))
    metrics.update(_aggregate_record_stats("label_prior", label_prior_records, ["label_prior_entropy", "label_prior_min", "label_prior_max"]))
    return metrics


def build_phase567_comparison_rows(run_summaries):
    rows = []
    if not run_summaries:
        return rows

    baseline = run_summaries[0].get("metrics", {})
    baseline_avg = baseline.get("average_accuracy")
    baseline_final = baseline.get("final_accuracy")
    baseline_pd = baseline.get("performance_degradation")

    for summary in run_summaries:
        metrics = summary.get("metrics", {})
        settings = summary.get("settings", {})
        run_name = summary.get("run_name", "")
        avg = metrics.get("average_accuracy")
        final = metrics.get("final_accuracy")
        pd = metrics.get("performance_degradation")
        row = {
            "run_name": run_name,
            "label": ABLATION_LABELS.get(run_name, run_name),
            "status": summary.get("status", "unknown"),
            "phase5_transform": settings.get("phase5_transform"),
            "phase6_alignment": settings.get("phase6_alignment"),
            "label_prior": settings.get("label_prior_enabled"),
            "phase7_mode": settings.get("phase7_mode"),
            "phase7_source": settings.get("phase7_source"),
            "transductive": settings.get("transductive"),
            "average_accuracy": avg,
            "final_accuracy": final,
            "performance_degradation": pd,
            "base_class_final_accuracy": metrics.get("base_class_final_accuracy"),
            "novel_class_final_accuracy": metrics.get("novel_class_final_accuracy"),
            "phase5_common_direction_norm": metrics.get("phase5_common_direction_norm_mean"),
            "phase5_whitening_var_mean": metrics.get("phase5_whitening_var_mean_mean"),
            "phase5_lda_dim_used": metrics.get("phase5_lda_dim_used_mean"),
            "phase6_ot_cost_mean": metrics.get("phase6_ot_cost_mean_mean"),
            "phase6_weight_entropy": metrics.get("phase6_weight_entropy_mean"),
            "label_prior_entropy": metrics.get("label_prior_label_prior_entropy_mean"),
            "phase7_pairwise_similarity_mean": metrics.get("phase7_pairwise_similarity_mean_mean"),
            "phase7_affected_prototypes": metrics.get("phase7_affected_prototypes_mean"),
            "phase7_displacement_norm_mean": metrics.get("phase7_displacement_norm_mean_mean"),
            "delta_avg": _delta(avg, baseline_avg),
            "delta_final": _delta(final, baseline_final),
            "delta_pd": _delta(baseline_pd, pd),
            "suspicious_drop": _is_suspicious_drop(_delta(avg, baseline_avg), _delta(final, baseline_final)),
            "error": summary.get("error"),
            "is_baseline": summary is run_summaries[0],
        }
        rows.append(row)
    return rows


def phase567_settings(cfg):
    opts = cfg.TRAINER.BiMC
    p5 = opts.PHASE5
    p6 = opts.PHASE6
    p7 = opts.PHASE7
    return {
        "fusion_beta_mode": opts.FUSION_BETA_MODE,
        "fusion_geometry": opts.FUSION_GEOMETRY,
        "dataset_beta": cfg.DATASET.BETA,
        "phase5_enabled": bool(p5.ENABLED),
        "phase5_transform": p5.SPACE_TRANSFORM,
        "phase5_apply_to": p5.APPLY_TO,
        "phase6_enabled": bool(p6.ENABLED),
        "phase6_alignment": p6.ALIGNMENT_MODE,
        "label_prior_enabled": bool(p6.LABEL_PRIOR_ENABLED or p6.ALIGNMENT_MODE == "label_prior_correction"),
        "label_prior_mode": p6.LABEL_PRIOR_MODE,
        "transductive": bool(p6.ENABLED and (p6.LABEL_PRIOR_ENABLED or p6.ALIGNMENT_MODE == "label_prior_correction")),
        "phase7_enabled": bool(p7.ENABLED),
        "phase7_mode": p7.SEPARATION_MODE,
        "phase7_source": p7.REPULSION_SOURCE,
    }


def phase567_run_dir(cfg, dataset_name):
    output_dir = _cfg_get(cfg, "OUTPUT_DIR", "results/phase567")
    timestamp = _cfg_get(cfg, "PHASE567_TIMESTAMP", "") or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = _cfg_get(cfg, "RUN_NAME", "")
    path = os.path.join(output_dir, _safe_name(dataset_name), timestamp)
    if run_name:
        path = os.path.join(path, _safe_name(run_name))
    return path


def _render_run_markdown(dataset_name, cfg, summary, comparison_rows, notes):
    metrics = summary.get("metrics", {})
    settings = summary.get("settings", {})
    lines = [
        "# Phase 5/6/7 Report",
        "",
        f"- Date/time: {datetime.now().isoformat(timespec='seconds')}",
        f"- Dataset: {dataset_name}",
        f"- Backbone: {cfg.MODEL.BACKBONE.NAME}",
        f"- Run: {summary.get('run_name', '')}",
        f"- Status: {summary.get('status', 'unknown')}",
        "",
        "## Config",
        "",
        "| Setting | Value |",
        "| --- | --- |",
    ]
    for key in [
        "fusion_beta_mode",
        "fusion_geometry",
        "dataset_beta",
        "phase5_transform",
        "phase5_apply_to",
        "phase6_alignment",
        "label_prior_enabled",
        "label_prior_mode",
        "phase7_mode",
        "phase7_source",
        "transductive",
    ]:
        lines.append(f"| {key} | {settings.get(key)} |")

    if summary.get("error"):
        lines.extend(["", "## Error", "", str(summary["error"])])

    lines.extend([
        "",
        "## Accuracy",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Average accuracy | {_fmt(metrics.get('average_accuracy'))} |",
        f"| Final accuracy | {_fmt(metrics.get('final_accuracy'))} |",
        f"| Performance degradation | {_fmt(metrics.get('performance_degradation'))} |",
        f"| Base final accuracy | {_fmt(metrics.get('base_class_final_accuracy'))} |",
        f"| Novel final accuracy | {_fmt(metrics.get('novel_class_final_accuracy'))} |",
        "",
        "## Phase Statistics",
        "",
    ])
    lines.extend(_phase_stats_table(metrics))
    lines.extend([
        "",
        "## Session Accuracy",
        "",
    ])
    lines.extend(_session_table(summary.get("session_metrics", [])))

    lines.extend(["", "## Baseline Comparison", ""])
    if comparison_rows:
        lines.extend(_comparison_table(comparison_rows))
    else:
        lines.append("Baseline comparison is unavailable for this standalone run.")

    if settings.get("transductive"):
        lines.extend([
            "",
            "## Transductive Warning",
            "",
            "This run uses unlabeled test-batch or session predictions for label-prior correction. It does not use test labels, but it is transductive.",
        ])

    lines.extend(["", "## Interpretation", ""])
    lines.extend(_interpretation(comparison_rows or [summary_to_row(summary)]))

    if notes:
        lines.extend(["", "## Notes", ""])
        for note in notes:
            lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _render_aggregate_markdown(dataset_name, cfg, rows):
    lines = [
        "# Phase 5/6/7 Ablation Report",
        "",
        f"- Date/time: {datetime.now().isoformat(timespec='seconds')}",
        f"- Dataset: {dataset_name}",
        f"- Backbone: {cfg.MODEL.BACKBONE.NAME}",
        "",
        "## Config",
        "",
        "| Setting | Value |",
        "| --- | --- |",
        f"| Seed | {cfg.SEED} |",
        f"| Fusion beta mode | fixed |",
        f"| Fusion geometry | linear |",
        f"| Dataset beta | {cfg.DATASET.BETA} |",
        "",
        "## Comparison",
        "",
    ]
    lines.extend(_comparison_table(rows))
    lines.extend(["", "## Phase 5 Stats", ""])
    lines.extend(_phase5_stats_table(rows))
    lines.extend(["", "## Phase 6 Stats", ""])
    lines.extend(_phase6_stats_table(rows))
    lines.extend(["", "## Phase 7 Stats", ""])
    lines.extend(_phase7_stats_table(rows))
    transductive = [row for row in rows if row.get("transductive")]
    if transductive:
        lines.extend([
            "",
            "## Transductive Warning",
            "",
            "Label-prior rows use unlabeled test predictions for prior estimation. They do not use test labels, but comparisons should mark them as transductive.",
            "",
            "Transductive methods: " + ", ".join(row["run_name"] for row in transductive),
        ])
    lines.extend(["", "## Interpretation", ""])
    lines.extend(_interpretation(rows))
    lines.append("")
    return "\n".join(lines)


def summary_to_row(summary):
    return build_phase567_comparison_rows([summary])[0]


def _comparison_table(rows):
    if not rows:
        return ["No runs were recorded."]
    lines = [
        "| Run | Status | P5 | P6 | Prior | P7 | Transductive | Avg | Final | PD | Delta Avg | Delta Final | Base Final | Novel Final | Suspicious |",
        "| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.get('run_name')} | {row.get('status')} | {row.get('phase5_transform')} | "
            f"{row.get('phase6_alignment')} | {row.get('label_prior')} | "
            f"{row.get('phase7_mode')}:{row.get('phase7_source')} | {row.get('transductive')} | "
            f"{_fmt(row.get('average_accuracy'))} | {_fmt(row.get('final_accuracy'))} | "
            f"{_fmt(row.get('performance_degradation'))} | {_fmt(row.get('delta_avg'))} | "
            f"{_fmt(row.get('delta_final'))} | {_fmt(row.get('base_class_final_accuracy'))} | "
            f"{_fmt(row.get('novel_class_final_accuracy'))} | {row.get('suspicious_drop')} |"
        )
    return lines


def _session_table(session_metrics):
    if not session_metrics:
        return ["No session metrics were recorded."]
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


def _phase_stats_table(metrics):
    lines = [
        "| Phase | Statistic | Value |",
        "| --- | --- | ---: |",
        f"| Phase 5 | common_direction_norm_mean | {_fmt(metrics.get('phase5_common_direction_norm_mean'))} |",
        f"| Phase 5 | whitening_var_mean_mean | {_fmt(metrics.get('phase5_whitening_var_mean_mean'))} |",
        f"| Phase 5 | lda_dim_used_mean | {_fmt(metrics.get('phase5_lda_dim_used_mean'))} |",
        f"| Phase 6 | ot_cost_mean_mean | {_fmt(metrics.get('phase6_ot_cost_mean_mean'))} |",
        f"| Phase 6 | alignment_weight_entropy_mean | {_fmt(metrics.get('phase6_weight_entropy_mean'))} |",
        f"| Phase 6 | label_prior_entropy_mean | {_fmt(metrics.get('label_prior_label_prior_entropy_mean'))} |",
        f"| Phase 7 | pairwise_similarity_mean | {_fmt(metrics.get('phase7_pairwise_similarity_mean_mean'))} |",
        f"| Phase 7 | affected_prototypes_mean | {_fmt(metrics.get('phase7_affected_prototypes_mean'))} |",
        f"| Phase 7 | displacement_norm_mean | {_fmt(metrics.get('phase7_displacement_norm_mean_mean'))} |",
    ]
    return lines


def _phase5_stats_table(rows):
    return _phase_table(
        rows,
        [
            ("CDR norm", "phase5_common_direction_norm"),
            ("Whiten var mean", "phase5_whitening_var_mean"),
            ("LDA dim", "phase5_lda_dim_used"),
        ],
    )


def _phase6_stats_table(rows):
    return _phase_table(
        rows,
        [
            ("OT cost mean", "phase6_ot_cost_mean"),
            ("Weight entropy", "phase6_weight_entropy"),
            ("Prior entropy", "label_prior_entropy"),
        ],
    )


def _phase7_stats_table(rows):
    return _phase_table(
        rows,
        [
            ("Pairwise sim", "phase7_pairwise_similarity_mean"),
            ("Affected", "phase7_affected_prototypes"),
            ("Displacement", "phase7_displacement_norm_mean"),
        ],
    )


def _phase_table(rows, columns):
    if not rows:
        return ["No rows were recorded."]
    header = "| Run | " + " | ".join(label for label, _ in columns) + " |"
    divider = "| --- | " + " | ".join("---:" for _ in columns) + " |"
    lines = [header, divider]
    for row in rows:
        values = " | ".join(_fmt(row.get(key)) for _, key in columns)
        lines.append(f"| {row.get('run_name')} | {values} |")
    return lines


def _interpretation(rows):
    if not rows:
        return ["- No completed runs are available."]
    successful = [row for row in rows if row.get("status") == "success" and row.get("average_accuracy") is not None]
    if not successful:
        return ["- All requested runs failed or produced no metrics."]
    baseline = rows[0]
    best_avg = max(successful, key=lambda row: _metric_value(row.get("average_accuracy")))
    best_final = max(successful, key=lambda row: _metric_value(row.get("final_accuracy")))
    suspicious = [row for row in rows if row.get("suspicious_drop")]
    return [
        f"- Space transformation helped: {_helped(rows, baseline, lambda row: row.get('phase5_transform') not in (None, 'none'))}.",
        f"- OT/support-aware text helped: {_helped(rows, baseline, lambda row: row.get('phase6_alignment') in ('ot_text_image', 'support_aware_text'))}.",
        f"- Label-prior correction helped: {_helped(rows, baseline, lambda row: row.get('label_prior'))}.",
        f"- Prototype separation helped: {_helped(rows, baseline, lambda row: row.get('phase7_mode') not in (None, 'none'))}.",
        f"- Best method by average: {best_avg['run_name']} ({_fmt(best_avg.get('average_accuracy'))}).",
        f"- Best method by final: {best_final['run_name']} ({_fmt(best_final.get('final_accuracy'))}).",
        f"- Reject methods with huge drops: {', '.join(row['run_name'] for row in suspicious) if suspicious else 'none detected'}.",
    ]


def _helped(rows, baseline, predicate):
    if baseline.get("average_accuracy") is None:
        return "needs original baseline"
    candidates = [
        row for row in rows
        if predicate(row) and row.get("status") == "success" and row.get("average_accuracy") is not None
    ]
    if not candidates:
        return "needs more tests"
    best = max(candidates, key=lambda row: _metric_value(row.get("average_accuracy")))
    avg_delta = _delta(best.get("average_accuracy"), baseline.get("average_accuracy"))
    final_delta = _delta(best.get("final_accuracy"), baseline.get("final_accuracy"))
    verdict = "yes" if avg_delta is not None and final_delta is not None and avg_delta > 0 and final_delta >= 0 else "no"
    return f"{verdict} ({best['run_name']}, delta_avg={_fmt(avg_delta)}, delta_final={_fmt(final_delta)})"


def _aggregate_record_stats(prefix, records, keys):
    output = {}
    for key in keys:
        values = [float(record[key]) for record in records if record.get(key) is not None]
        stats = _value_stats(values)
        for stat_name, value in stats.items():
            output[f"{prefix}_{key}_{stat_name}"] = value
    return output


def _records_with_run_names(run_summaries, key):
    records = []
    for summary in run_summaries:
        run_name = summary.get("run_name", "")
        for record in summary.get(key, []):
            row = dict(record)
            row["run_name"] = run_name
            records.append(row)
    return records


def _write_session_accuracy_csv(path, session_metrics):
    rows = []
    for session_id, record in enumerate(session_metrics):
        rows.append({
            "session": session_id,
            "mean_acc": record.get("mean_acc"),
            "base_avg_acc": record.get("base_avg_acc"),
            "inc_avg_acc": record.get("inc_avg_acc"),
            "harmonic_acc": record.get("harmonic_acc"),
            "task_acc": json.dumps(record.get("task_acc", [])),
        })
    _write_csv(path, rows)


def _write_combined_session_accuracy_csv(path, run_summaries):
    rows = []
    for summary in run_summaries:
        for session_id, record in enumerate(summary.get("session_metrics", [])):
            rows.append({
                "run_name": summary.get("run_name", ""),
                "session": session_id,
                "mean_acc": record.get("mean_acc"),
                "base_avg_acc": record.get("base_avg_acc"),
                "inc_avg_acc": record.get("inc_avg_acc"),
                "harmonic_acc": record.get("harmonic_acc"),
                "task_acc": json.dumps(record.get("task_acc", [])),
            })
    _write_csv(path, rows)


def _write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(_json_safe(value), handle, indent=2)


def _run_manifest(cfg, dataset_name, summary):
    return {
        "dataset": dataset_name,
        "run_name": summary.get("run_name"),
        "status": summary.get("status"),
        "error": summary.get("error"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": cfg_to_dict(cfg),
    }


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
        return json.dumps(_json_safe(value))
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


def _metric_value(value):
    return float("-inf") if value is None else float(value)


def _delta(current, baseline):
    if current is None or baseline is None:
        return None
    return _round_or_none(float(current) - float(baseline))


def _is_suspicious_drop(delta_avg, delta_final, threshold=-5.0):
    return (
        delta_avg is not None and delta_avg <= threshold
    ) or (
        delta_final is not None and delta_final <= threshold
    )
