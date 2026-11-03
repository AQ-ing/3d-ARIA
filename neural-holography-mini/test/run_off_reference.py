"""Manual off-only comparison at the original training resolution."""
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def main():
    test = Path(__file__).resolve().parent
    run = test / "experiments" / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_off_reference")
    run.mkdir(parents=True)
    results = []
    for name in ("current", "reference"):
        command = [sys.executable, "-B", "-u", str(test / "train_single.py"),
                   "--image", "data/image/0031.png", "--controlnet-mode", "off",
                   "--size", "1920", "1072", "--preprocess", "homography",
                   "--loss-region", "roi", "--loss", "vgg-low", "--seed", "42",
                   "--learning-rate", "0.001", "--steps", "1000",
                   "--output", str(run / "output" / name),
                   "--save-model", str(run / "checkpoint" / (name + ".pth"))]
        if name == "reference":
            command.append("--off-reference")
        record = {"name": name, "command": command, "initialization": "random seed 42"}
        results.append(record)
        (run / "manifest.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print("Running " + name + "; log: " + str(run / (name + ".log")), flush=True)
        with (run / (name + ".log")).open("w", encoding="utf-8") as log:
            record["returncode"] = subprocess.run(command, cwd=test.parent, stdout=log,
                                                   stderr=subprocess.STDOUT).returncode
        text = (run / (name + ".log")).read_text(encoding="utf-8")
        for metric in ("amplitude_psnr", "reference_fixed_scale_psnr"):
            match = re.search(metric + r"=([\d.]+)", text)
            record[metric] = float(match[1]) if match else None
        (run / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Finished: " + str(run / "results.json"), flush=True)


if __name__ == "__main__":
    main()
