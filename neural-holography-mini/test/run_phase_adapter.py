"""Manual batch: two adapter learning rates plus a zero-base ablation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime
from console_mode import uninterrupted_console


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--suite", choices=("initial", "diagnosis", "refiner"), default="initial")
    args = parser.parse_args()
    test = Path(__file__).resolve().parent
    jobs = [("adapter_lr1e3", "small", 1e-3, False), ("adapter_lr1e4", "small", 1e-4, False),
            ("zero_base_lr1e3", "small", 1e-3, True)]
    steps = 600
    if args.suite == "diagnosis":
        steps = 1200
        jobs = [("pixel_lr003", "pixel", 0.03, False), ("pixel_lr01", "pixel", 0.1, False),
                ("small_1200", "small", 1e-3, False), ("wide_1200", "wide", 1e-3, False),
                ("context_1200", "context", 1e-3, False)]
    if args.suite == "refiner":
        steps = 200
        jobs = [("fixed8", "fixed8", 1e-3, False), ("fixed16", "fixed16", 1e-3, False),
                ("learned8", "learned8", 1e-3, False), ("learned16", "learned16", 1e-3, False)]
    for path in (test / "experiments/20261102_102239_off_reference/checkpoint/current.pth",
                 test.parent / "data/image/0031.png", test.parent / "data/depth/0031.png"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.dry_run:
        print(jobs)
        print(f"{steps} outer training steps (fixed baselines: 0); 1920x1072; ROI 1600x880; MSE. No training started.")
        return
    run = test / "experiments" / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_phase_adapter_" + args.suite)
    run.mkdir(parents=True, exist_ok=False)
    records = []
    for name, kind, lr, zero_base in jobs:
        job_steps = 0 if kind.startswith("fixed") else steps
        command = [sys.executable, "-B", "-u", str(test / "train_phase_adapter.py"),
                   "--output", str(run / name), "--steps", str(job_steps), "--learning-rate", str(lr),
                   "--adapter", kind]
        if zero_base:
            command.append("--without-base")
        row = dict(name=name, command=command, status="running")
        records.append(row)
        (run / "results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"Starting {name}; output={run / name}", flush=True)
        with (run / (name + ".log")).open("w", encoding="utf-8") as log:
            with subprocess.Popen(command, cwd=test.parent, stdin=subprocess.DEVNULL,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace", bufsize=1,
                                  env={**os.environ, "PYTHONIOENCODING": "utf-8"}) as process:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                row["returncode"] = process.wait()
        row["status"] = "completed" if row["returncode"] == 0 else "failed"
        metrics = run / name / "results.json"
        if metrics.is_file():
            row["metrics"] = json.loads(metrics.read_text(encoding="utf-8"))
        (run / "results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    lines = ["# Phase adapter results", "", f"{steps} outer training steps (fixed baselines: 0); full-resolution ROI amplitude MSE.",
             "Historical ControlNet-only MSE: 26.89 dB at 1200 steps. Pixel runs are per-image diagnostics, not reusable networks.", "",
             "| Run | Status | PSNR | Parameters | Elapsed seconds | Evaluation seconds | Peak allocated MB |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in records:
        m = row.get("metrics", {})
        lines.append(f"| {row['name']} | {row['status']} | {m.get('psnr')} | {m.get('trainable_parameters')} | {m.get('elapsed_seconds')} | {m.get('evaluation_seconds')} | {m.get('peak_allocated_mb')} |")
    (run / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("Completed: " + str(run), flush=True)


if __name__ == "__main__":
    with uninterrupted_console():
        main()
