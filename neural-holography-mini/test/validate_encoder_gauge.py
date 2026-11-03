"""Check a fixed physical carrier alignment before the frozen phase encoder."""
import json
import math
from datetime import datetime
from pathlib import Path
from train_single import (ROOT, HoloNet, ImageLoader, cv2, np, torch, propagation_asm,
                          load_holonet_checkpoint, crop_image, resize_keep_aspect, pad_crop_to_res)


def main():
    test = Path(__file__).resolve().parent
    run = test / "experiments" / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_encoder_gauge")
    run.mkdir(parents=True, exist_ok=False)
    output = run / "output"
    output.mkdir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size, roi = (1072, 1920), (880, 1600)
    wavelength, pitch, z0 = 520e-9, 6.4e-6, 0.2
    checkpoint = test / "experiments/20261102_102239_off_reference/checkpoint/current.pth"
    target = next(iter(ImageLoader(str(ROOT / "data/image/0031.png"), channel=1,
        image_res=size, homography_res=roi, shuffle=False,
        vertical_flips=False, horizontal_flips=False)))[0].to(device)
    model = HoloNet().to(device).eval()
    load_holonet_checkpoint(model, checkpoint, device)
    t = crop_image(target, roi, stacked_complex=False)

    def metric(amp):
        scale = (amp*t).mean()/amp.square().mean().clamp_min(1e-12)
        mse = (amp*scale-t).square().mean()
        return {"psnr": -10*np.log10(max(mse.item(), 1e-20)), "mse": mse.item(), "scale": scale.item()}

    def save(name, amp, scale=1):
        pixels = np.uint8((amp*scale).clamp(0, 1).squeeze().cpu().numpy()*255)
        if not cv2.imwrite(str(output / (name+".png")), pixels):
            raise OSError(name)

    def encode(field):
        phase = model.final_phase_only(torch.cat((field.abs(), field.angle()), 1))
        return torch.polar(torch.ones_like(phase), phase)

    def render(field, z):
        return crop_image(propagation_asm(field, pitch, wavelength, z).abs(), roi, stacked_complex=False)

    def rotation(z):
        phase_offset = 2*math.pi*(z-z0)/wavelength
        return complex(math.cos(phase_offset), math.sin(phase_offset))

    results = {"config": {"checkpoint": str(checkpoint), "training_steps": 0,
        "rotation": "exp(+i*2*pi*(z-z0)/wavelength) before frozen final_phase_only",
        "size": size, "roi": roi, "wavelength": wavelength, "pitch": pitch, "base_depth": z0}}
    with torch.inference_mode():
        save("target", t)
        object_field = torch.polar(target, model.initial_phase(target))
        results["depths"] = []
        for z in (0.18, 0.19, 0.2, 0.21, 0.22):
            desired = propagation_asm(object_field, pitch, wavelength, -z)
            row = {"depth": z}
            for label, field in (("raw", desired), ("carrier_aligned", desired*rotation(z))):
                amp = render(encode(field), z)
                row[label] = metric(amp)
                save(f"{label}_{z}", amp, row[label]["scale"])
            results["depths"].append(row)
        raw = cv2.imread(str(ROOT / "data/depth/0031.png"), cv2.IMREAD_GRAYSCALE)
        dep = pad_crop_to_res(resize_keep_aspect(raw[None].astype(np.float32)/255, roi), size)
        near = (torch.from_numpy(np.ascontiguousarray(dep))[None].to(device)>0.1).float()
        masks, distances = (near, 1-near), (0.18, 0.22)
        layers = [propagation_asm(object_field*m, pitch, wavelength, -z) for m, z in zip(masks, distances)]
        aligned = [u*rotation(z) for u, z in zip(layers, distances)]
        results["multiplane"] = {}
        for label, field in (("raw_sum", sum(layers)), ("aligned_sum", sum(aligned))):
            for encoding in ("complex", "frozen_encoder"):
                actual = field if encoding == "complex" else encode(field)
                amplitudes = [render(actual, z) for z in distances]
                merged = sum(a*crop_image(m, roi, stacked_complex=False) for a, m in zip(amplitudes, masks))
                row = metric(merged)
                name = label+"_"+encoding
                for region, amp, m in zip(("near", "far"), amplitudes, masks):
                    mask = crop_image(m, roi, stacked_complex=False)
                    mse = ((amp*row["scale"]-t).square()*mask).sum()/mask.sum()
                    row[region+"_psnr"] = -10*np.log10(max(mse.item(), 1e-20))
                    save(name+"_"+region, amp, row["scale"])
                save(name+"_merged", merged, row["scale"])
                results["multiplane"][name] = row
    (run / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(str(run), flush=True)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
