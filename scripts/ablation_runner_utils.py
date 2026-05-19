import contextlib
import copy
import csv
import json
import os
import shutil
import sys
import traceback
from datetime import datetime
from time import perf_counter


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def load_config(path):
    """Load JSON/YAML-like config files without requiring PyYAML at import time."""
    with open(path, "r") as handle:
        text = handle.read()

    if path.endswith(".json"):
        return json.loads(text)

    try:
        import yaml

        loaded = yaml.safe_load(text)
        return loaded or {}
    except ModuleNotFoundError:
        return _parse_simple_yaml(text)


def deep_update(base, overrides):
    """Return a deep-copied mapping with recursive overrides applied."""
    result = copy.deepcopy(base)
    for key, value in (overrides or {}).items():
        if (
            isinstance(value, dict)
            and isinstance(result.get(key), dict)
        ):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def save_config_snapshot(config, out_path):
    directory = os.path.dirname(out_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(out_path, "w") as handle:
        json.dump(to_plain(config), handle, indent=2)


def run_eval_with_config(config_path_or_obj, output_dir, dataset, device, gpu_id, seed=None):
    """Run the existing evaluation pipeline and write normalized per-run artifacts."""
    os.makedirs(output_dir, exist_ok=True)
    stdout_path = os.path.join(output_dir, "stdout.log")
    stderr_path = os.path.join(output_dir, "stderr.log")
    phase_output_dir = os.path.join(output_dir, "_phase_outputs")
    os.makedirs(phase_output_dir, exist_ok=True)

    start = perf_counter()
    summary = None
    raw_metrics = {}
    status = "success"
    error_message = ""

    with open(stdout_path, "w") as stdout, open(stderr_path, "w") as stderr:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                cfg = _coerce_executable_config(config_path_or_obj, dataset)
                _apply_runtime_overrides(cfg, phase_output_dir, device, gpu_id, seed)
                save_config_snapshot(cfg, os.path.join(output_dir, "config.json"))

                from engine.engine import Runner
                from utils.util import set_gpu, set_seed

                set_seed(cfg.SEED)
                if gpu_id is not None:
                    set_gpu(gpu_id)

                runner = Runner(cfg)
                summary = runner.run()
                raw_metrics = summary.get("metrics") or _metrics_from_session_metrics(
                    summary.get("session_metrics", [])
                )
                _copy_phase_outputs(summary.get("run_dir"), output_dir)
            except Exception as exc:
                status = "failed"
                error_message = str(exc)
                traceback.print_exc()

    duration_sec = perf_counter() - start

    if not os.path.exists(os.path.join(output_dir, "config.json")):
        save_config_snapshot(config_path_or_obj, os.path.join(output_dir, "config.json"))

    if status == "failed":
        _write_json(os.path.join(output_dir, "metrics.json"), raw_metrics)
        _write_empty_session_csv(os.path.join(output_dir, "session_accuracy.csv"))
        _write_failure_report(
            os.path.join(output_dir, "report.md"),
            dataset=dataset,
            error_message=error_message,
            duration_sec=duration_sec,
        )
    else:
        _write_json(os.path.join(output_dir, "metrics.json"), raw_metrics)
        if summary is not None and not os.path.exists(os.path.join(output_dir, "session_accuracy.csv")):
            _write_session_accuracy_csv(
                os.path.join(output_dir, "session_accuracy.csv"),
                summary.get("session_metrics", []),
            )
        if not os.path.exists(os.path.join(output_dir, "report.md")):
            _write_success_report(
                os.path.join(output_dir, "report.md"),
                dataset=dataset,
                metrics=raw_metrics,
                duration_sec=duration_sec,
            )

    return {
        "status": status,
        "duration_sec": round(duration_sec, 3),
        "output_dir": output_dir,
        "report_path": os.path.join(output_dir, "report.md"),
        "raw_metrics": raw_metrics,
        "error_message": error_message,
    }


def collect_metrics(output_dir):
    direct = os.path.join(output_dir, "metrics.json")
    if os.path.isfile(direct):
        with open(direct, "r") as handle:
            return json.load(handle)

    for root, _, files in os.walk(output_dir):
        if "metrics.json" in files:
            with open(os.path.join(root, "metrics.json"), "r") as handle:
                return json.load(handle)
    return {}


def normalize_metric_row(raw_metrics):
    raw_metrics = raw_metrics or {}
    return {
        "average_accuracy": _first_present(raw_metrics, "average_accuracy", "avg_accuracy", "avg"),
        "final_accuracy": _first_present(raw_metrics, "final_accuracy", "last_accuracy", "final"),
        "performance_degradation": _first_present(raw_metrics, "performance_degradation", "pd"),
        "base_final_accuracy": _first_present(
            raw_metrics,
            "base_final_accuracy",
            "base_class_final_accuracy",
            "base_avg_acc",
        ),
        "novel_final_accuracy": _first_present(
            raw_metrics,
            "novel_final_accuracy",
            "novel_class_final_accuracy",
            "inc_avg_acc",
        ),
    }


def to_plain(value):
    if isinstance(value, dict):
        return {key: to_plain(item) for key, item in value.items()}
    if hasattr(value, "items"):
        return {key: to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


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


def _coerce_executable_config(config_path_or_obj, dataset):
    if not isinstance(config_path_or_obj, str):
        return config_path_or_obj

    from main import setup_cfg

    data_cfg = resolve_dataset_cfg(dataset)
    train_cfg = resolve_train_cfg(config_path_or_obj)
    return setup_cfg(data_cfg, train_cfg)


def _apply_runtime_overrides(cfg, output_dir, device, gpu_id, seed):
    cfg.defrost()
    cfg.OUTPUT_DIR = output_dir
    if device is not None:
        cfg.DEVICE.DEVICE_NAME = device
    if gpu_id is not None:
        cfg.DEVICE.GPU_ID = gpu_id
    if seed is not None:
        cfg.SEED = seed
    cfg.freeze()


def _copy_phase_outputs(run_dir, output_dir):
    if not run_dir or not os.path.isdir(run_dir):
        return

    copies = {
        "metrics.json": "metrics.json",
        "session_accuracy.csv": "session_accuracy.csv",
        "phase1_report.md": "report.md",
        "report.md": "report.md",
        "beta_stats.csv": "beta_stats.csv",
        "beta_by_class.csv": "beta_by_class.csv",
    }
    for source_name, target_name in copies.items():
        source = os.path.join(run_dir, source_name)
        if os.path.isfile(source):
            shutil.copy2(source, os.path.join(output_dir, target_name))

    stats_dir = os.path.join(output_dir, "stats")
    for name in os.listdir(run_dir):
        if not (name.endswith(".csv") or name.endswith(".json")):
            continue
        if name in copies:
            continue
        os.makedirs(stats_dir, exist_ok=True)
        shutil.copy2(os.path.join(run_dir, name), os.path.join(stats_dir, name))


def _metrics_from_session_metrics(session_metrics):
    session_acc = [float(item["mean_acc"]) for item in session_metrics if "mean_acc" in item]
    final_accuracy = session_acc[-1] if session_acc else None
    average_accuracy = sum(session_acc) / len(session_acc) if session_acc else None
    performance_degradation = session_acc[0] - session_acc[-1] if len(session_acc) >= 2 else None
    last_session = session_metrics[-1] if session_metrics else {}
    return {
        "session_accuracy": session_acc,
        "final_accuracy": _round_or_none(final_accuracy),
        "average_accuracy": _round_or_none(average_accuracy),
        "performance_degradation": _round_or_none(performance_degradation),
        "base_class_final_accuracy": _round_or_none(last_session.get("base_avg_acc")),
        "novel_class_final_accuracy": _round_or_none(last_session.get("inc_avg_acc")),
    }


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


def _write_empty_session_csv(path):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["session", "mean_acc", "base_avg_acc", "inc_avg_acc", "harmonic_acc", "task_acc"])


def _write_success_report(path, dataset, metrics, duration_sec):
    lines = [
        "# Final Combo Run",
        "",
        f"- Date/time: {datetime.now().isoformat(timespec='seconds')}",
        f"- Dataset: {dataset}",
        f"- Status: success",
        f"- Duration sec: {duration_sec:.3f}",
        "",
        "## Metrics",
        "",
        f"- Average accuracy: {_fmt(metrics.get('average_accuracy'))}",
        f"- Final accuracy: {_fmt(metrics.get('final_accuracy'))}",
        f"- Performance degradation: {_fmt(metrics.get('performance_degradation'))}",
        f"- Base final accuracy: {_fmt(metrics.get('base_class_final_accuracy'))}",
        f"- Novel final accuracy: {_fmt(metrics.get('novel_class_final_accuracy'))}",
        "",
    ]
    with open(path, "w") as handle:
        handle.write("\n".join(lines))


def _write_failure_report(path, dataset, error_message, duration_sec):
    lines = [
        "# Final Combo Run",
        "",
        f"- Date/time: {datetime.now().isoformat(timespec='seconds')}",
        f"- Dataset: {dataset}",
        f"- Status: failed",
        f"- Duration sec: {duration_sec:.3f}",
        "",
        "## Error",
        "",
        error_message or "Unknown error.",
        "",
    ]
    with open(path, "w") as handle:
        handle.write("\n".join(lines))


def _write_json(path, value):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(to_plain(value), handle, indent=2)


def _first_present(mapping, *keys):
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _round_or_none(value, digits=4):
    if value is None:
        return None
    return round(float(value), digits)


def _fmt(value):
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _parse_simple_yaml(text):
    root = {}
    stack = [(-1, root)]
    for raw_line in text.splitlines():
        line = _strip_yaml_comment(raw_line).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            raise ValueError(f"Unsupported YAML line: {raw_line}")
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if value == "":
            node = {}
            parent[key] = node
            stack.append((indent, node))
        else:
            parent[key] = _parse_yaml_scalar(value)
    return root


def _strip_yaml_comment(line):
    in_single = False
    in_double = False
    escaped = False
    chars = []
    for char in line:
        if escaped:
            chars.append(char)
            escaped = False
            continue
        if char == "\\":
            chars.append(char)
            escaped = True
            continue
        if char == "'" and not in_double:
            in_single = not in_single
            chars.append(char)
            continue
        if char == '"' and not in_single:
            in_double = not in_double
            chars.append(char)
            continue
        if char == "#" and not in_single and not in_double:
            break
        chars.append(char)
    return "".join(chars)


def _parse_yaml_scalar(value):
    if value.startswith(("'", '"')) and value.endswith(("'", '"')):
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_yaml_scalar(item.strip()) for item in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
