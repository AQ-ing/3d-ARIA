"""Re-evaluate the saved 960x640 experiments; no training or parameter updates."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from train_single import HoloNet, DepthControlNet, ImageLoader, amplitude_loss, propagation_asm


def main():
    test = Path(__file__).resolve().parent
    root = test.parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = next(iter(ImageLoader(str(root / "data/image/0031.png"), channel=1,
        image_res=(640, 960), direct_resize=True, shuffle=False,
        vertical_flips=False, horizontal_flips=False,
        depth_path=str(root / "data/depth/0031.png"))))
    target, depth = loaded[0].to(device), loaded[3].to(device)
    masks = ((depth > 0.1).float(), (depth <= 0.1).float())
    runs = [
        ("off_1000", "off_mse_960x640_1000.pth", False),
        ("off_refined", "off_refine_1000.pth", False),
        ("joint_1200", "joint_refined_off_1200.pth", True),
        ("control_only_1200", "control_refined_off_1200.pth", True),
    ]
    results, images = {}, {"target": target.cpu().squeeze().numpy()}
    with torch.inference_mode():
        for name, filename, use_depth in runs:
            checkpoint = torch.load(test / "checkpoint" / filename, map_location=device, weights_only=True)
            model = HoloNet().to(device).eval()
            model.load_state_dict(checkpoint["holonet"])
            control = DepthControlNet(model.final_phase_only.net[0]).to(device).eval() if use_depth else None
            if control is not None:
                control.load_state_dict(checkpoint["controlnet"])
            phase = model(target, depth=depth if use_depth else None, controlnet=control)
            field = torch.polar(torch.ones_like(phase), phase)
            if use_depth:
                planes = [propagation_asm(field, model.feature_size, model.wavelength, z).abs()
                          for z in (0.18, 0.22)]
                recon = sum(plane * mask for plane, mask in zip(planes, masks))
            else:
                recon = propagation_asm(field, model.feature_size, model.wavelength, 0.2).abs()
            mse, scale = amplitude_loss(recon, target, torch.nn.functional.mse_loss)
            error = (recon * scale - target).square()
            metrics = {"mse": mse.item(), "psnr": (-10 * mse.log10()).item()}
            for label, mask in zip(("near_region", "far_region"), masks):
                region_mse = (error * mask).sum() / mask.sum()
                metrics[label + "_psnr"] = (-10 * region_mse.log10()).item()
            results[name] = metrics
            images[name] = (recon * scale).cpu().squeeze().numpy()
            print(name, metrics, flush=True)
    (test / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    # Equal amplitude range and identical crops for a fair visual comparison.
    names = ["target", "off_refined", "joint_1200", "control_only_1200"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for column, name in enumerate(names):
        img = images[name]
        axes[0, column].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[0, column].set_title(name + (f"\n{results[name]['psnr']:.2f} dB" if name in results else ""))
        axes[1, column].imshow(img[40:260, 350:650], cmap="gray", vmin=0, vmax=1)
        for row in range(2):
            axes[row, column].axis("off")
    fig.tight_layout()
    fig.savefig(test / "output/comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
