"""No-training validation of complex-field depth transport and phase encoding."""
import json
from datetime import datetime
from pathlib import Path
import torch.nn.functional as F
from train_single import (ROOT, HoloNet, ImageLoader, cv2, np, torch, propagation_asm,
                          load_holonet_checkpoint, crop_image, resize_keep_aspect, pad_crop_to_res)


def main():
    test = Path(__file__).resolve().parent
    run = test / "experiments" / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_depth_physics")
    run.mkdir(parents=True, exist_ok=False)
    output = run / "output"
    output.mkdir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size, roi = (1072, 1920), (880, 1600)
    wavelength, pitch, z0 = 520e-9, 6.4e-6, 0.2
    target = next(iter(ImageLoader(str(ROOT / "data/image/0031.png"), channel=1,
        image_res=size, homography_res=roi, shuffle=False, vertical_flips=False, horizontal_flips=False)))[0].to(device)
    model = HoloNet().to(device).eval()
    initial = test / "experiments/20261102_102239_off_reference/checkpoint/current.pth"
    load_holonet_checkpoint(model, initial, device)
    results = {"config": {"size": size, "roi": roi, "wavelength": wavelength, "pitch": pitch,
                           "base_depth": z0, "checkpoint": str(initial), "training_steps": 0}}
    cropped = lambda x: crop_image(x, roi, stacked_complex=False)
    target_roi = cropped(target)

    def metrics(amp):
        scale = (amp * target_roi).mean() / amp.square().mean().clamp_min(1e-12)
        mse = (amp * scale - target_roi).square().mean().item()
        return {"mse": mse, "psnr": -10 * np.log10(max(mse, 1e-20)), "scale": scale.item()}

    def save(name, amp, scale):
        pixels = np.uint8((amp * scale).clamp(0, 1).squeeze().cpu().numpy() * 255)
        if not cv2.imwrite(str(output / (name + ".png")), pixels):
            raise OSError(name)

    def relative(a, b):
        return ((a-b).abs().square().sum() / b.abs().square().sum().clamp_min(1e-30)).sqrt().item()

    def old(field, z):
        return propagation_asm(field, pitch, wavelength, z)

    with torch.inference_mode():
        phi0 = model(target)
        slm0 = torch.polar(torch.ones_like(phi0), phi0)
        baseline = cropped(old(slm0, z0).abs())
        results["baseline_current"] = metrics(baseline)
        save("target", target_roi, 1)
        save("baseline", baseline, results["baseline_current"]["scale"])
        # 1. Common enlarged FFT grid: no intermediate crop, stable carrier-free kernel.
        h, w = size
        padded = F.pad(slm0, (w//2, w//2, h//2, h//2))
        fy = torch.fft.fftfreq(2*h, d=pitch, dtype=torch.float64, device=device)[:, None]
        fx = torch.fft.fftfreq(2*w, d=pitch, dtype=torch.float64, device=device)[None, :]
        f2 = fx.square() + fy.square()
        root = (1 / wavelength**2 - f2).sqrt()
        q = -2 * np.pi * f2 / (root + 1 / wavelength)
        del root, f2

        def stable(field, z, band=None):
            kernel = torch.polar(torch.ones_like(q), q * z).to(torch.complex64)
            if band is not None:
                kernel = kernel * band
            return torch.fft.ifft2(torch.fft.fft2(field, norm="ortho") * kernel, norm="ortho")

        common_band = ((fx.abs() < 1/(wavelength*np.sqrt(1+(0.44/(2*w*pitch))**2))) &
                       (fy.abs() < 1/(wavelength*np.sqrt(1+(0.44/(2*h*pitch))**2)))).float()
        band_input = torch.fft.ifft2(torch.fft.fft2(padded, norm="ortho") * common_band, norm="ortho")
        stable_base = stable(band_input, z0, common_band)
        stable_crop = stable_base[..., h//2:h//2+h, w//2:w//2+w]
        results["stable_common_band_base"] = metrics(cropped(stable_crop.abs()))
        results["ideal_common_grid"] = []
        for z in (0.18, 0.19, 0.21, 0.22):
            shifted = stable(band_input, z0-z, common_band)
            ideal = stable(shifted, z, common_band)
            intermediate = shifted[..., h//2:h//2+h, w//2:w//2+w]
            aperture = F.pad(intermediate, (w//2, w//2, h//2, h//2))
            finite = stable(aperture, z, common_band)
            row = {"depth": z, "complex_relative_error_no_crop": relative(ideal, stable_base),
                   "complex_relative_error_with_aperture": relative(finite, stable_base),
                   "intermediate_energy_outside_slm_fraction":
                       1 - (intermediate.abs().square().sum() / shifted.abs().square().sum()).item()}
            results["ideal_common_grid"].append(row)
        del padded, band_input, stable_base, stable_crop, shifted, ideal, intermediate, aperture, finite, q, common_band
        # 2. Actual code: translate a trained phase-only output, then keep/discard amplitude.
        results["output_field_transfer"] = []
        for z in (0.18, 0.19, 0.21, 0.22):
            shifted = old(slm0, z0-z)
            complex_amp = cropped(old(shifted, z).abs())
            phase_amp = cropped(old(torch.polar(torch.ones_like(phi0), shifted.angle()), z).abs())
            row = {"depth": z, "unshifted": metrics(cropped(old(slm0, z).abs())),
                   "shift_complex": metrics(complex_amp), "shift_then_phase_only": metrics(phase_amp)}
            results["output_field_transfer"].append(row)
            if z in (0.18, 0.22):
                save(f"output_shift_complex_{z}", complex_amp, row["shift_complex"]["scale"])
                save(f"output_shift_phase_{z}", phase_amp, row["shift_then_phase_only"]["scale"])
        # 3. Reuse both existing neural components, changing only the physical depth.
        object_field = torch.polar(target, model.initial_phase(target))
        results["reuse_encoder_single_depth"] = []
        for z in (0.18, 0.19, 0.2, 0.21, 0.22):
            desired = old(object_field, -z)
            phase = model.final_phase_only(torch.cat((desired.abs(), desired.angle()), 1))
            amp = cropped(old(torch.polar(torch.ones_like(phase), phase), z).abs())
            row = {"depth": z, **metrics(amp)}
            results["reuse_encoder_single_depth"].append(row)
            save(f"reuse_encoder_{z}", amp, row["scale"])
        # 4. Two layers: isolate propagation, coherent cross-talk, and phase encoding.
        raw = cv2.imread(str(ROOT / "data/depth/0031.png"), cv2.IMREAD_GRAYSCALE)
        dep = pad_crop_to_res(resize_keep_aspect(raw[None].astype(np.float32)/255, roi), size)
        near = (torch.from_numpy(np.ascontiguousarray(dep))[None].to(device) > 0.1).float()
        masks = (near, 1-near)
        distances = (0.18, 0.22)
        layers = [old(object_field * m, -z) for m, z in zip(masks, distances)]
        isolated = [old(u, z) for u, z in zip(layers, distances)]
        summed = layers[0] + layers[1]
        complex_planes = [old(summed, z) for z in distances]
        phase_angle = summed.angle()
        phase_encoded = model.final_phase_only(torch.cat((summed.abs(), summed.angle()), 1))
        results["multiplane"] = {}
        for name, fields in [("isolated_layers_not_simultaneous", isolated), ("summed_complex", complex_planes),
                             ("summed_angle", [old(torch.polar(torch.ones_like(phase_angle), phase_angle), z) for z in distances]),
                             ("summed_frozen_encoder", [old(torch.polar(torch.ones_like(phase_encoded), phase_encoded), z) for z in distances])]:
            amplitudes = [cropped(u.abs()) for u in fields]
            merged = sum(a * cropped(m) for a, m in zip(amplitudes, masks))
            row = metrics(merged)
            for label, a, m in zip(("near", "far"), amplitudes, masks):
                err = (a * row["scale"] - target_roi).square()
                region_mse = (err * cropped(m)).sum() / cropped(m).sum()
                row[label+"_psnr"] = -10*np.log10(max(region_mse.item(), 1e-20))
                save(name+"_"+label, a, row["scale"])
            results["multiplane"][name] = row
            save(name+"_merged", merged, row["scale"])
        results["complex_crosstalk"] = []
        for i, (m, z) in enumerate(zip(masks, distances)):
            unwanted = old(layers[1-i], z)
            desired = isolated[i]
            row = {"depth": z,
                   "unwanted_to_desired_energy_in_region":
                       ((cropped(unwanted).abs().square()*cropped(m)).sum() /
                        (cropped(desired).abs().square()*cropped(m)).sum()).item()}
            results["complex_crosstalk"].append(row)
    (run / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(str(run), flush=True)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
