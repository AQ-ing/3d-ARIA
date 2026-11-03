"""Manual sequential ablations; all runs start from the same off checkpoint."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime
from console_mode import uninterrupted_console


TEST = Path(__file__).resolve().parent
INITIAL = TEST / "experiments/20261102_102239_off_reference/checkpoint/current.pth"
EXPERIMENTS = [
    ("same_plane", "controlnet-only", "vgg-low", 1e-4, (0.2, 0.2), 300),
    ("joint_vgg", "joint", "vgg-low", 1e-4, (0.18, 0.22), 1200),
    ("only_mse", "controlnet-only", "mse", 1e-4, (0.18, 0.22), 1200),
    ("joint_mse", "joint", "mse", 1e-4, (0.18, 0.22), 1200),
    ("only_small_gap", "controlnet-only", "vgg-low", 1e-4, (0.195, 0.205), 1200),
    ("only_lr3e4", "controlnet-only", "vgg-low", 3e-4, (0.18, 0.22), 1200),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print plan without training")
    args = parser.parse_args()
    for path in (INITIAL, TEST.parent / "data/image/0031.png", TEST.parent / "data/depth/0031.png"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.dry_run:
        for experiment in EXPERIMENTS:
            print(experiment)
        print("Inputs exist. No training started.")
        return
    run = TEST / "experiments" / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_control_ablation")
    run.mkdir(parents=True, exist_ok=False)
    records = []
    for name, mode, loss, lr, distances, steps in EXPERIMENTS:
        command = [sys.executable, "-B", "-u", str(TEST / "train_single.py"),
                   "--image", "data/image/0031.png", "--depth", "data/depth/0031.png",
                   "--init-checkpoint", str(INITIAL), "--controlnet-mode", mode,
                   "--size", "1920", "1072", "--preprocess", "homography", "--loss-region", "roi",
                   "--channel", "1", "--seed", "42", "--loss", loss, "--learning-rate", str(lr),
                   "--steps", str(steps), "--distance-near", str(distances[0]),
                   "--distance-far", str(distances[1]), "--depth-threshold", "0.1", "--near-is", "bright",
                   "--output", str(run / "output" / name),
                   "--save-model", str(run / "checkpoint" / (name + ".pth"))]
        if name == "same_plane":
            command.append("--same-plane-diagnostic")
        record = dict(name=name, mode=mode, loss=loss, learning_rate=lr, distances=distances,
                      steps=steps, command=command, status="running")
        records.append(record)
        (run / "results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"Starting {name} ({len(records)}/{len(EXPERIMENTS)})", flush=True)
        with (run / (name + ".log")).open("w", encoding="utf-8") as log:
            with subprocess.Popen(command, cwd=TEST.parent, stdin=subprocess.DEVNULL,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                  encoding="utf-8", errors="replace", bufsize=1,
                                  env={**os.environ, "PYTHONIOENCODING": "utf-8"}) as process:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                record["returncode"] = process.wait()
        record["status"] = "completed" if record["returncode"] == 0 else "failed"
        text = (run / (name + ".log")).read_text(encoding="utf-8")
        for metric in ("amplitude_mse", "amplitude_psnr"):
            match = re.search(metric + r"=([\d.eE+-]+)", text)
            record[metric] = float(match[1]) if match else None
        for region in ("near", "far"):
            match = re.search(region + r":.*psnr=([\d.]+)", text)
            record[region + "_psnr"] = float(match[1]) if match else None
        (run / "results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    lines = ["# ControlNet ablation", "", "Baseline: off 31.128668 dB; prior control-only vgg 23.801595 dB.",
             "same_plane is a diagnostic, not a two-depth result.", "",
             "| Run | Steps | PSNR | Near | Far | Status |", "| --- | ---: | ---: | ---: | ---: | --- |"]
    for r in records:
        lines.append(f"| {r['name']} | {r['steps']} | {r['amplitude_psnr']} | {r['near_psnr']} | {r['far_psnr']} | {r['status']} |")
    (run / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("Finished: " + str(run), flush=True)


if __name__ == "__main__":
    with uninterrupted_console():
        main()
