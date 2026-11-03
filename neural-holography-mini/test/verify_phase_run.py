"""Recompute saved phase-adapter metrics without optimization."""
import argparse
import json
from pathlib import Path
from train_single import (ROOT, torch, np, cv2, ImageLoader, HoloNet, load_holonet_checkpoint,
                          resize_keep_aspect, pad_crop_to_res, crop_image, propagation_asm, amplitude_loss)
from phase_adapter import build_adapter
from learned_refiner import LearnedPhaseRefiner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    if not run.is_relative_to(Path(__file__).resolve().parent):
        parser.error("Run must be under test/")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(run / "checkpoint/adapter.pth", map_location=device, weights_only=True)
    c = ckpt["config"]
    target = next(iter(ImageLoader(str(ROOT / "data/image/0031.png"), channel=c["channel"],
                      image_res=c["size"], homography_res=c["roi"], shuffle=False,
                      vertical_flips=False, horizontal_flips=False)))[0].to(device)
    depth = cv2.imread(str(ROOT / "data/depth/0031.png"), cv2.IMREAD_GRAYSCALE)
    depth = pad_crop_to_res(resize_keep_aspect(depth[None].astype(np.float32) / 255, c["roi"]), c["size"])
    near = (torch.from_numpy(np.ascontiguousarray(depth))[None].to(device) > c["threshold"]).float()
    model = HoloNet(wavelength=c["wavelength"], feature_size=c["pitch"], distance=c["base_distance"]).to(device).eval()
    load_holonet_checkpoint(model, c["initial"], device)
    kind = c.get("adapter_kind", "small")
    refinement = kind.startswith(("learned", "fixed"))
    adapter = (LearnedPhaseRefiner() if refinement else build_adapter(kind, target.shape)).to(device).eval()
    adapter.load_state_dict(ckpt["phase_adapter"])
    roi_target = crop_image(target, c["roi"], stacked_complex=False)

    def reconstruct(phase):
        field = torch.polar(torch.ones_like(phase), phase)
        planes = [propagation_asm(field, c["pitch"], c["wavelength"], z).abs() for z in c["distances"]]
        return crop_image(near * planes[0] + (1-near) * planes[1], c["roi"], stacked_complex=False)

    def objective(phase):
        return amplitude_loss(reconstruct(phase), roi_target, torch.nn.functional.mse_loss)[0]

    with torch.no_grad():
        base = model(target)
        if c["without_base"]:
            base = torch.zeros_like(base)
        offset = (near * c["distances"][0] + (1-near) * c["distances"][1] - c["base_distance"]) / c["distance_normalizer"]
        if refinement:
            phase = adapter.refine(base, objective, 16 if kind.endswith("16") else 8,
                                   learned=kind.startswith("learned"))
        else:
            phase = base + adapter(adapter.condition(target, base, near, offset))
        mse, scale = amplitude_loss(reconstruct(phase), roi_target, torch.nn.functional.mse_loss)
        metrics = {"mse": mse.item(), "psnr": -10*np.log10(mse.item()), "scale": scale.item()}
    saved = json.loads((run / "results.json").read_text(encoding="utf-8"))
    metrics["psnr_difference_from_saved"] = metrics["psnr"] - saved["psnr"]
    (run / "verified_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
