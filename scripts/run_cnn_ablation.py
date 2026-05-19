import argparse
import csv
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

CNN_MODES = [
    "none",
    "cnn_proto_adjust",
    "cnn_query_beta",
    "cnn_proto_adjust_plus_query_beta",
]
PROTO_ADJUST_MODES = {"cnn_proto_adjust", "cnn_proto_adjust_plus_query_beta"}
DATASET_CONFIGS = {
    "cub200": "configs/datasets/cub200.yaml",
    "cifar100": "configs/datasets/cifar100.yaml",
}
TRAINER_CONFIGS = {
    "bimc": "configs/trainers/bimc.yaml",
    "bimc_ensemble": "configs/trainers/bimc_ensemble.yaml",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run CNN BiMC ablations.")
    parser.add_argument("--dataset", choices=["cub200", "cifar100", "both"], default="both")
    parser.add_argument("--trainer", choices=["bimc", "bimc_ensemble"], default="bimc")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--cnn-backbone", default="resnet50")
    parser.add_argument("--cnn-lambda-grid", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.20])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default="outputs/cnn_ablation")
    return parser.parse_args()


def selected_datasets(dataset):
    if dataset == "both":
        return ["cub200", "cifar100"]
    return [dataset]


def lambda_values_for_mode(mode, lambda_grid):
    if mode in PROTO_ADJUST_MODES:
        return lambda_grid
    return [None]


def lambda_tag(value):
    if value is None:
        return ""
    return f"_lambda{value:.2f}".replace(".", "p")


def run_name(dataset, trainer, seed, mode, lambda_value):
    return f"{dataset}_{trainer}_seed{seed}_{mode}{lambda_tag(lambda_value)}"


def build_command(dataset, trainer, seed, mode, lambda_value, args):
    use_cnn = "false" if mode == "none" else "true"
    command = [
        "python",
        "main.py",
        "--data_cfg",
        DATASET_CONFIGS[dataset],
        "--train_cfg",
        TRAINER_CONFIGS[trainer],
        "--seed",
        str(seed),
        "--use_cnn_branch",
        use_cnn,
        "--cnn_experiment_mode",
        mode,
        "--cnn_backbone",
        args.cnn_backbone,
        "--cnn_topk",
        "5",
        "--cnn_projection",
        "random_orthogonal",
        "--cnn_cache_features",
        "true",
        "--save_phase1_report",
        "true",
        "--output_dir",
        args.output_dir,
        "--run_name",
        run_name(dataset, trainer, seed, mode, lambda_value),
    ]
    if lambda_value is not None:
        command.extend(["--cnn_lambda", f"{lambda_value:.6g}"])
    return command


def command_to_string(command):
    return " ".join(shlex.quote(part) for part in command)


def extract_final_acc(log_path):
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except OSError:
        return ""
    for line in reversed(lines):
        if line.startswith("Final acc:"):
            return line.strip()
    return ""


def write_summary(summary_path, rows):
    fieldnames = [
        "dataset",
        "trainer",
        "seed",
        "mode",
        "cnn_lambda",
        "returncode",
        "duration_sec",
        "log_path",
        "final_acc",
        "command",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    commands = []
    for dataset in selected_datasets(args.dataset):
        for seed in args.seeds:
            for mode in CNN_MODES:
                for lambda_value in lambda_values_for_mode(mode, args.cnn_lambda_grid):
                    command = build_command(dataset, args.trainer, seed, mode, lambda_value, args)
                    commands.append((dataset, seed, mode, lambda_value, command))

    if args.dry_run:
        for _, _, _, _, command in commands:
            print(command_to_string(command))
        return 0

    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "cnn_ablation_summary.csv")

    rows = []
    for dataset, seed, mode, lambda_value, command in commands:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = run_name(dataset, args.trainer, seed, mode, lambda_value)
        log_path = os.path.join(log_dir, f"{timestamp}_{name}.log")
        command_text = command_to_string(command)
        print(f"Running: {command_text}")
        started = time.time()
        with open(log_path, "w", encoding="utf-8") as log_handle:
            log_handle.write(f"$ {command_text}\n")
            log_handle.flush()
            process = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        duration = time.time() - started
        rows.append({
            "dataset": dataset,
            "trainer": args.trainer,
            "seed": seed,
            "mode": mode,
            "cnn_lambda": "" if lambda_value is None else f"{lambda_value:.6g}",
            "returncode": process.returncode,
            "duration_sec": f"{duration:.2f}",
            "log_path": log_path,
            "final_acc": extract_final_acc(log_path),
            "command": command_text,
        })
        write_summary(summary_path, rows)
        print(f"Finished returncode={process.returncode}, log={log_path}")

    print(f"Summary CSV: {summary_path}")
    return 0 if all(int(row["returncode"]) == 0 for row in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
