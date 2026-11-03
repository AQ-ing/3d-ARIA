"""Manually run one ControlNet-only experiment from the 31.13 dB off weights."""
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def main():
    test = Path(__file__).resolve().parent
    root = test.parent
    initial = test / "experiments/20261102_102239_off_reference/checkpoint/current.pth"
    if not initial.is_file():
        raise FileNotFoundError(initial)
    run = test / "experiments" / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_control_from_current")
    run.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-B", "-u", str(test / "train_single.py"),
               "--image", "data/image/0031.png", "--depth", "data/depth/0031.png",
               "--controlnet-mode", "controlnet-only", "--init-checkpoint", str(initial),
               "--size", "1920", "1072", "--preprocess", "homography",
               "--loss-region", "roi", "--loss", "vgg-low", "--channel", "1",
               "--seed", "42", "--learning-rate", "0.0001", "--steps", "1200",
               "--distance-near", "0.18", "--distance-far", "0.22",
               "--depth-threshold", "0.1", "--near-is", "bright",
               "--output", str(run / "output"),
               "--save-model", str(run / "checkpoint/controlnet-only.pth")]
    record = {"name": "controlnet-only_from_current", "initial": str(initial),
              "command": command, "steps": 1200, "learning_rate": 1e-4,
              "baseline_off_roi_psnr": 31.128668}
    (run / "manifest.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("Training log: " + str(run / "train.log"), flush=True)
    with (run / "train.log").open("w", encoding="utf-8") as log:
        with subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace", bufsize=1,
                              env={**os.environ, "PYTHONIOENCODING": "utf-8"}) as process:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            returncode = process.wait()
    record["returncode"] = returncode
    log = (run / "train.log").read_text(encoding="utf-8")
    for metric in ("amplitude_mse", "amplitude_psnr"):
        match = re.search(metric + r"=([\d.eE+-]+)", log)
        record[metric] = float(match[1]) if match else None
    for region in ("near", "far"):
        match = re.search(region + r":.*psnr=([\d.]+)", log)
        record[region + "_psnr"] = float(match[1]) if match else None
    (run / "results.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("Results: " + str(run / "results.json"), flush=True)
    if returncode:
        raise SystemExit(returncode)


if __name__ == "__main__":
    main()
