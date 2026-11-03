"""Single-image adapter experiment; the frozen base is evaluated only once."""
import argparse
import json
from pathlib import Path
import random
import time

from train_single import (ROOT, HoloNet, DepthControlNet, ImageLoader, cv2, np, torch,
                          load_holonet_checkpoint, propagation_asm, amplitude_loss,
                          resize_keep_aspect, pad_crop_to_res, crop_image)
from phase_adapter import build_adapter
from learned_refiner import LearnedPhaseRefiner


def main():
    # ==================== 1. Fixed experiment geometry and explicit run settings ====================
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Run directory under test/")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--adapter", choices=("small", "wide", "context", "pixel", "learned8", "learned16", "fixed8", "fixed16"), default="small")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--without-base", action="store_true", help="Ablation: zero base phase in both input and output")
    args = parser.parse_args()
    test = Path(__file__).resolve().parent
    run = args.output.resolve()
    refinement = args.adapter.startswith(("learned", "fixed"))
    fixed = args.adapter.startswith("fixed")
    if not run.is_relative_to(test) or args.steps < (0 if fixed else 1) or args.learning_rate <= 0:
        parser.error("Output must be under test/; steps and learning rate must be positive")
    run.mkdir(parents=True, exist_ok=False)
    if fixed and args.steps != 0:
        parser.error("Fixed update baseline requires --steps 0 (no training)")
    initial = test / "experiments/20261102_102239_off_reference/checkpoint/current.pth"
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size, roi = (1072, 1920), (880, 1600)
    distances, wavelength, pitch = (0.18, 0.22), 520e-9, 6.4e-6
    config = {"steps": args.steps, "learning_rate": args.learning_rate, "without_base": args.without_base,
              "initial": str(initial), "size": size, "roi": roi, "distances": distances,
              "base_distance": 0.2, "distance_normalizer": 0.02, "wavelength": wavelength,
              "pitch": pitch, "threshold": 0.1, "near_is": "bright", "seed": 42, "channel": 1,
              "loss": "merged_amplitude_mse", "adapter_version": 1, "adapter_kind": args.adapter, "device": str(device)}
    if refinement:
        config.update(refinement_steps=16 if args.adapter.endswith("16") else 8,
                      refinement_base_lr=0.1, beta1=0.9, beta2=0.999,
                      gradient_training="stop_gradient_first_order", shared_rule_parameters=74)
    (run / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    # ==================== 2. Match the existing RGB/depth geometry ====================
    loaded = next(iter(ImageLoader(str(ROOT / "data/image/0031.png"), channel=1,
                     image_res=size, homography_res=roi, shuffle=False,
                     vertical_flips=False, horizontal_flips=False)))
    target = loaded[0].to(device)
    raw_depth = cv2.imread(str(ROOT / "data/depth/0031.png"), cv2.IMREAD_GRAYSCALE)
    if raw_depth is None or raw_depth.shape != loaded[1][0]:
        raise ValueError("RGB/depth dimensions must match")
    depth = pad_crop_to_res(resize_keep_aspect(raw_depth[None].astype(np.float32) / 255, roi), size)
    near = (torch.from_numpy(np.ascontiguousarray(depth))[None].to(device) > 0.1).float()
    masks = [crop_image(m, roi, stacked_complex=False) for m in (near, 1 - near)]
    roi_target = crop_image(target, roi, stacked_complex=False)
    # ==================== 3. Freeze base, cache phase, and discard its computation graph ====================
    started = time.perf_counter()
    model = HoloNet().to(device).eval().requires_grad_(False)
    load_holonet_checkpoint(model, initial, device)
    with torch.no_grad():
        base_phase = model(target).detach()
    # Both runs use identical adapter initialization, regardless of base computation.
    adapter = (LearnedPhaseRefiner() if refinement else build_adapter(args.adapter, base_phase.shape)).to(device)
    control_parameters = sum(p.numel() for p in DepthControlNet(model.final_phase_only.net[0]).parameters())
    base_parameters = sum(p.numel() for p in model.parameters())
    del model
    if args.without_base:
        base_phase = torch.zeros_like(base_phase)
    offset = (near * distances[0] + (1 - near) * distances[1] - 0.2) / 0.02
    condition = adapter.condition(target, base_phase, near, offset).detach()
    # Cache forward ASM kernels; no changing optical settings during a run.
    field = torch.polar(torch.ones_like(base_phase), base_phase)
    kernels = [propagation_asm(field, pitch, wavelength, z, return_kernel=True).detach() for z in distances]
    optimizer = torch.optim.Adam(adapter.parameters(), lr=args.learning_rate)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    setup_seconds = time.perf_counter() - started
    trainable = 0 if fixed else sum(p.numel() for p in adapter.parameters())
    print(f"adapter_parameters={trainable} controlnet_parameters={control_parameters} frozen_base_parameters={base_parameters}", flush=True)

    def reconstruct_phase(phase):
        field = torch.polar(torch.ones_like(phase), phase)
        planes = [crop_image(propagation_asm(field, pitch, wavelength, z, kernel=k).abs(), roi,
                             stacked_complex=False) for z, k in zip(distances, kernels)]
        return sum(p * m for p, m in zip(planes, masks))

    def objective(phase):
        return amplitude_loss(reconstruct_phase(phase), roi_target, torch.nn.functional.mse_loss)[0]

    def reconstruct():
        if refinement:
            iterations = 16 if args.adapter.endswith("16") else 8
            phase = adapter.refine(base_phase, objective, iterations, learned=not fixed)
        else:
            phase = base_phase + adapter(condition)
        return reconstruct_phase(phase)

    # ==================== 4. Optimize only the adapter; measure quality at completed steps ====================
    history = []
    started = time.perf_counter()
    for step in range(args.steps + 1):
        if step:
            optimizer.zero_grad(set_to_none=True)
            loss, _ = amplitude_loss(reconstruct(), roi_target, torch.nn.functional.mse_loss)
            loss.backward()
            optimizer.step()
        if step == 0 or step % 100 == 0 or step == args.steps:
            if device.type == "cuda":
                torch.cuda.synchronize()
            evaluation_started = time.perf_counter()
            with torch.no_grad():
                recon = reconstruct()
                mse, scale = amplitude_loss(recon, roi_target, torch.nn.functional.mse_loss)
                psnr = -10 * np.log10(max(mse.item(), 1e-12))
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            evaluation_seconds = time.perf_counter() - evaluation_started
            row = dict(step=step, mse=mse.item(), psnr=psnr, elapsed_seconds=elapsed,
                       evaluation_seconds=evaluation_seconds)
            history.append(row)
            (run / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            print(f"step {step}/{args.steps}: mse={mse.item():.8f} psnr={psnr:.4f} elapsed={elapsed:.1f}s", flush=True)
    # ==================== 5. Store standalone adapter, optical settings, and final merged reconstruction ====================
    error = (recon * scale - roi_target).square()
    metrics = dict(history[-1], trainable_parameters=trainable, controlnet_parameters=control_parameters,
                   setup_seconds=setup_seconds,
                   peak_allocated_mb=torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None)
    for label, mask in zip(("near", "far"), masks):
        region_mse = (error * mask).sum() / mask.sum().clamp_min(1)
        metrics[label + "_psnr"] = -10 * np.log10(max(region_mse.item(), 1e-12))
    (run / "results.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run / "checkpoint").mkdir()
    torch.save({"phase_adapter": adapter.state_dict(), "config": config}, run / "checkpoint/adapter.pth")
    (run / "output").mkdir()
    pixels = np.uint8(np.clip((recon * scale).squeeze().cpu().numpy(), 0, 1) * 255)
    if not cv2.imwrite(str(run / "output/mergerecon0031.png"), pixels):
        raise OSError("Could not write reconstruction")
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
