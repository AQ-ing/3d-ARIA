"""No-training check of Fresnel lens-phase depth transfer using the current ASM."""
import json
from datetime import datetime
from pathlib import Path

import torch.nn.functional as F
from train_single import (ROOT, HoloNet, ImageLoader, cv2, np, torch,
                          propagation_asm, load_holonet_checkpoint, crop_image)


def main():
    test = Path(__file__).resolve().parent
    run = test / "experiments" / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_quadratic_depth")
    output = run / "output"
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size, roi = (1072, 1920), (880, 1600)
    wavelength, pitch, z0 = 520e-9, 6.4e-6, 0.2
    checkpoint = test / "experiments/20261102_102239_off_reference/checkpoint/current.pth"
    target = next(iter(ImageLoader(str(ROOT / "data/image/0031.png"), channel=1,
        image_res=size, homography_res=roi, shuffle=False,
        vertical_flips=False, horizontal_flips=False)))[0].to(device)
    model = HoloNet().to(device).eval()
    load_holonet_checkpoint(model, checkpoint, device)
    h, w = size
    yy, xx = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float64)-h//2,
                            torch.arange(w, device=device, dtype=torch.float64)-w//2, indexing="ij")
    radius_squared = (xx.square() + yy.square()) * pitch**2
    cropped = lambda a: crop_image(a, roi, stacked_complex=False)

    def warp(a, coordinate_scale):
        # Coordinates follow the centered FFT convention. Zero beyond the SLM grid.
        gx = (2*(xx*coordinate_scale+w//2)+1)/w-1
        gy = (2*(yy*coordinate_scale+h//2)+1)/h-1
        grid = torch.stack((gx, gy), -1).float()[None]
        return F.grid_sample(a, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    def metrics(a, t):
        a, t = cropped(a), cropped(t)
        scale = (a*t).mean() / a.square().mean().clamp_min(1e-12)
        mse = (a*scale-t).square().mean().item()
        return {"psnr": -10*np.log10(max(mse, 1e-20)), "mse": mse, "scale": scale.item()}

    def save(name, a, scale=1):
        pixels = np.uint8((cropped(a)*scale).clamp(0, 1).squeeze().cpu().numpy()*255)
        if not cv2.imwrite(str(output / (name+".png")), pixels):
            raise OSError(name)

    def render(phi, z, lens=None):
        if lens is not None:
            phi = phi + lens
        return propagation_asm(torch.polar(torch.ones_like(phi), phi), pitch, wavelength, z).abs()

    results = {"config": {"checkpoint": str(checkpoint), "size": size, "roi": roi,
        "wavelength": wavelength, "pitch": pitch, "base_depth": z0, "training_steps": 0,
        "metric": "ROI amplitude PSNR, optimal scalar; bilinear resampling for scaled references",
        "prewarp_amplitude": "A(m*r); m amplitude gain omitted to keep model input in original range; evaluation fits scalar"}}
    with torch.inference_mode():
        phi0 = model(target)
        baseline = render(phi0, z0)
        results["baseline"] = metrics(baseline, target)
        save("target", target)
        save("baseline", baseline, results["baseline"]["scale"])
        results["depths"] = []
        for z in (0.18, 0.19, 0.21, 0.22):
            m = z/z0
            lens = (np.pi/wavelength * (1/z0-1/z) * radius_squared).float()[None, None]
            actual = render(phi0, z, lens)
            scaled_target = warp(target, 1/m) / m
            scaled_baseline = warp(baseline, 1/m) / m
            prewarped_target = warp(target, m)
            prewarped_phi = model(prewarped_target)
            prewarped_actual = render(prewarped_phi, z, lens)
            prewarped_base = render(prewarped_phi, z0)
            row = {"depth": z, "magnification": m,
                   "lens_vs_scaled_target": metrics(actual, scaled_target),
                   "lens_vs_scaled_baseline": metrics(actual, scaled_baseline),
                   "prewarped_base_vs_prewarped_target": metrics(prewarped_base, prewarped_target),
                   "prewarp_then_lens_vs_original_target": metrics(prewarped_actual, target)}
            results["depths"].append(row)
            save(f"lens_{z}", actual, row["lens_vs_scaled_target"]["scale"])
            save(f"scaled_target_{z}", scaled_target)
            save(f"prewarp_lens_{z}", prewarped_actual, row["prewarp_then_lens_vs_original_target"]["scale"])
            print(json.dumps(row), flush=True)
    (run / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(str(run), flush=True)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
