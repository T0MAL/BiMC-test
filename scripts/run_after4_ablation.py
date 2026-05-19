import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description="Run all ablations after Phase 4.")
    parser.add_argument("--datasets", nargs="+", default=["cifar100", "cub200"])
    parser.add_argument("--config", default="configs/trainers/bimc.yaml")
    parser.add_argument("--output-dir", default="results/after4")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu-id", default="0;")
    parser.add_argument("--continue-on-error", dest="continue_on_error", action="store_true", default=True)
    parser.add_argument("--no-continue-on-error", dest="continue_on_error", action="store_false")
    parser.add_argument("--dry-run", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = os.path.join(args.output_dir, timestamp)
    phase567_dir = os.path.join(root_dir, "phase567")
    logs_dir = os.path.join(root_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    execution = []
    all_rows = []
    for dataset in args.datasets:
        dataset_key = _dataset_key(dataset)
        log_path = os.path.join(logs_dir, f"{dataset_key}.log")
        cmd = [
            sys.executable,
            os.path.join(ROOT, "scripts", "run_phase567_ablation.py"),
            "--dataset",
            dataset,
            "--config",
            args.config,
            "--output-dir",
            phase567_dir,
            "--device",
            args.device,
            "--gpu-id",
            args.gpu_id,
            "--timestamp",
            timestamp,
        ]
        if not args.continue_on_error:
            cmd.append("--no-continue-on-error")

        status = "dry_run"
        returncode = None
        if args.dry_run:
            with open(log_path, "w") as handle:
                handle.write("DRY RUN\n")
                handle.write(" ".join(cmd) + "\n")
        else:
            with open(log_path, "w") as handle:
                proc = subprocess.run(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
            returncode = proc.returncode
            status = "success" if proc.returncode == 0 else "failed"
            if proc.returncode != 0 and not args.continue_on_error:
                execution.append(_execution_record(dataset, cmd, status, returncode, log_path))
                break

        execution.append(_execution_record(dataset, cmd, status, returncode, log_path))
        metrics_path = os.path.join(phase567_dir, dataset_key, timestamp, "aggregate_phase567_metrics.json")
        if os.path.isfile(metrics_path):
            rows = _load_dataset_rows(metrics_path, dataset_key)
            all_rows.extend(rows)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "root_dir": root_dir,
        "phase567_dir": phase567_dir,
        "continue_on_error": bool(args.continue_on_error),
        "dry_run": bool(args.dry_run),
        "execution": execution,
    }
    _write_json(os.path.join(root_dir, "combined_after4_metrics.json"), {"execution": execution, "comparison": all_rows})
    _write_csv(os.path.join(root_dir, "combined_after4_metrics.csv"), all_rows)
    _write_json(os.path.join(root_dir, "run_manifest.json"), manifest)
    report = _render_combined_report(execution, all_rows)
    with open(os.path.join(root_dir, "combined_after4_report.md"), "w") as handle:
        handle.write(report)

    print(f"Combined after-Phase-4 report: {os.path.join(root_dir, 'combined_after4_report.md')}")


def _execution_record(dataset, cmd, status, returncode, log_path):
    return {
        "dataset": dataset,
        "status": status,
        "returncode": returncode,
        "command": " ".join(cmd),
        "log_path": log_path,
    }


def _load_dataset_rows(metrics_path, dataset_key):
    with open(metrics_path, "r") as handle:
        payload = json.load(handle)
    rows = []
    for row in payload.get("comparison", []):
        new_row = dict(row)
        new_row["dataset"] = dataset_key
        rows.append(new_row)
    return rows


def _render_combined_report(execution, rows):
    lines = [
        "# After Phase 4 Ablation Report",
        "",
        f"- Date/time: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Execution Status",
        "",
        "| Dataset | Status | Return Code | Log |",
        "| --- | --- | ---: | --- |",
    ]
    for record in execution:
        lines.append(
            "| "
            f"{record['dataset']} | {record['status']} | {record['returncode']} | {record['log_path']} |"
        )

    lines.extend(["", "## Best Method By Dataset", ""])
    lines.extend(_best_by_dataset_table(rows))
    lines.extend(["", "## Best Method Overall", ""])
    best = _best_row(rows)
    if best:
        lines.append(
            f"- Best average accuracy: {best['dataset']} / {best['run_name']} "
            f"({_fmt(best.get('average_accuracy'))}, delta={_fmt(best.get('delta_avg'))})."
        )
    else:
        lines.append("- No successful metrics were found.")

    lines.extend(["", "## Comparison Against Original", ""])
    lines.extend(_original_comparison_table(rows))

    suspicious = [row for row in rows if row.get("suspicious_drop")]
    lines.extend(["", "## Suspicious Drops", ""])
    if suspicious:
        for row in suspicious:
            lines.append(
                f"- {row['dataset']} / {row['run_name']}: "
                f"delta_avg={_fmt(row.get('delta_avg'))}, delta_final={_fmt(row.get('delta_final'))}"
            )
    else:
        lines.append("- None detected.")

    transductive = [row for row in rows if row.get("transductive")]
    lines.extend(["", "## Transductive Methods", ""])
    if transductive:
        for row in transductive:
            lines.append(f"- {row['dataset']} / {row['run_name']}")
    else:
        lines.append("- None.")

    lines.append("")
    return "\n".join(lines)


def _best_by_dataset_table(rows):
    if not rows:
        return ["No metrics were loaded."]
    datasets = sorted({row["dataset"] for row in rows})
    lines = [
        "| Dataset | Best Avg Run | Avg | Delta Avg | Best Final Run | Final | Delta Final |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for dataset in datasets:
        dataset_rows = [row for row in rows if row["dataset"] == dataset and row.get("status") == "success"]
        if not dataset_rows:
            lines.append(f"| {dataset} | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        best_avg = max(dataset_rows, key=lambda row: _metric_value(row.get("average_accuracy")))
        best_final = max(dataset_rows, key=lambda row: _metric_value(row.get("final_accuracy")))
        lines.append(
            "| "
            f"{dataset} | {best_avg['run_name']} | {_fmt(best_avg.get('average_accuracy'))} | "
            f"{_fmt(best_avg.get('delta_avg'))} | {best_final['run_name']} | "
            f"{_fmt(best_final.get('final_accuracy'))} | {_fmt(best_final.get('delta_final'))} |"
        )
    return lines


def _original_comparison_table(rows):
    if not rows:
        return ["No metrics were loaded."]
    lines = [
        "| Dataset | Run | Avg | Final | Delta Avg | Delta Final | Transductive | Suspicious |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['dataset']} | {row['run_name']} | {_fmt(row.get('average_accuracy'))} | "
            f"{_fmt(row.get('final_accuracy'))} | {_fmt(row.get('delta_avg'))} | "
            f"{_fmt(row.get('delta_final'))} | {row.get('transductive')} | {row.get('suspicious_drop')} |"
        )
    return lines


def _best_row(rows):
    candidates = [row for row in rows if row.get("status") == "success" and row.get("average_accuracy") is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda row: _metric_value(row.get("average_accuracy")))


def _dataset_key(dataset):
    base = os.path.basename(dataset).lower().replace(".yaml", "")
    base = base.replace("-", "").replace("_", "")
    aliases = {
        "cub": "cub200",
        "cub200": "cub200",
        "cifar": "cifar100",
        "cifar100": "cifar100",
    }
    return aliases.get(base, base)


def _write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(value, handle, indent=2)


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


def _csv_value(value):
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value)
    return value


def _metric_value(value):
    return float("-inf") if value is None else float(value)


def _fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    return f"{float(value):.3f}"


if __name__ == "__main__":
    main()
