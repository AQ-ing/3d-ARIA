"""Run the complete experiment queue and collect results once it finishes."""
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def main():
    test = Path(__file__).resolve().parent
    root = test.parent
    run = test / "experiments" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run.mkdir(parents=True, exist_ok=False)
    records = []

    def execute(name, mode, initial, lr, steps, loss):
        weights = run / "checkpoint" / (name + ".pth")
        command = [sys.executable, "-B", "-u", str(test / "train_single.py"),
            "--image", "data/image/0031.png", "--depth", "data/depth/0031.png",
            "--size", "960", "640", "--loss-region", "full", "--loss", loss,
            "--controlnet-mode", mode, "--init-checkpoint", str(initial),
            "--learning-rate", str(lr), "--steps", str(steps),
            "--output", str(run / "output" / name), "--save-model", str(weights)]
        record = dict(name=name, mode=mode, initial=str(initial), lr=lr, steps=steps, loss=loss,
                      weights=str(weights), command=command)
        print("Running " + name, flush=True)
        with (run / (name + ".log")).open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        record["returncode"] = result.returncode
        records.append(record)
        # Preserve execution state even if a later job fails.
        (run / "manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        return record

    def score(record):
        text = (run / (record["name"] + ".log")).read_text(encoding="utf-8")
        match = re.search(r"amplitude_psnr=([\d.]+)", text)
        return float(match[1]) if record["returncode"] == 0 and match else None

    initial = root / "checkpoint/20261028_142836_541774_off_epoch_040.pth"
    if not initial.is_file():
        raise FileNotFoundError(initial)
    for loss in ("mse", "vgg-low"):
        base = execute("off_" + loss, "off", initial, 1e-4, 800, loss)
        if score(base) is None:
            print("Skipping dependent control runs: " + base["name"], flush=True)
            continue
        for mode in ("joint", "controlnet-only"):
            execute(mode + "_" + loss, mode, Path(base["weights"]), 1e-4, 1200, loss)
    # Collect all final metrics together, after the queue has finished.
    for record in records:
        record["amplitude_psnr"] = score(record)
        log = (run / (record["name"] + ".log")).read_text(encoding="utf-8")
        for region in ("near", "far"):
            match = re.search(region + r":.*psnr=([\d.]+)", log)
            record[region + "_psnr"] = float(match[1]) if match else None
    (run / "results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    lines = ["# Batch experiment results", "", "960x640, green amplitude, full-image evaluation, seed 42.",
             "MSE versus vgg-low; both off runs use the same dataset-pretrained checkpoint.",
             "Control modes start from their matching-loss off weights; steps exclude pretraining.", "",
             "| Run | Additional steps | PSNR (dB) | Near | Far | Status |",
             "| --- | ---: | ---: | ---: | ---: | --- |"]
    for r in records:
        lines.append(f"| {r['name']} | {r['steps']} | {r['amplitude_psnr']} | {r['near_psnr']} | {r['far_psnr']} | {r['returncode']} |")
    (run / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Completed. Report: " + str(run / "report.md"), flush=True)
    print(json.dumps([{k: r[k] for k in ("name", "amplitude_psnr", "returncode")} for r in records]), flush=True)


if __name__ == "__main__":
    main()
