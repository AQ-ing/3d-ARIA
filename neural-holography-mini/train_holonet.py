import argparse
import csv
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from holonet import HoloNet
from propagation import propagation_asm


def load_target(path, channel, height, width, device):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read input image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image[..., channel].astype(np.float32) / 255.0
    image = np.where(image <= 0.04045, image / 12.92, ((image + 0.055) / 1.055) ** 2.4)
    image = np.sqrt(image)
    tensor = torch.from_numpy(image)[None, None]
    tensor = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
    return tensor.to(device)


def save_phase(phase, path):
    phase = phase.detach().squeeze().cpu().numpy()
    phase_u8 = np.round((1.0 - ((phase + np.pi) % (2 * np.pi)) / (2 * np.pi)) * 255).astype(np.uint8)
    if not cv2.imwrite(path, phase_u8):
        raise OSError(f"Could not write phase image: {path}")


def main():
    parser = argparse.ArgumentParser(description="Overfit HoloNet to one target image.")
    parser.add_argument("--image", default="data/1.png")
    parser.add_argument("--channel", type=int, choices=(0, 1, 2), default=1, help="RGB channel: 0, 1, or 2")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--size", type=int, nargs=2, default=(1920, 1080), metavar=("WIDTH", "HEIGHT"),
                        help="Output resolution as width height (default: 1920 1080)")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--output", default="output")
    args = parser.parse_args()

    width, height = args.size
    if args.steps < 1 or min(width, height) < 16:
        parser.error("--steps must be positive and both output dimensions must be at least 16")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wavelength = (638e-9, 520e-9, 450e-9)[args.channel]
    target = load_target(args.image, args.channel, height, width, device)
    padded_height = (height + 15) // 16 * 16
    padded_width = (width + 15) // 16 * 16
    network_target = F.pad(target, (0, padded_width - width, 0, padded_height - height), mode="replicate")
    model = HoloNet(wavelength=wavelength).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    os.makedirs(args.output, exist_ok=True)

    losses = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        phase = model(network_target)[..., :height, :width]
        slm_field = torch.polar(torch.ones_like(phase), phase)
        recon = propagation_asm(slm_field, model.feature_size, wavelength, model.distance)
        recon_amp = recon.abs()
        scale = (recon_amp * target).mean() / recon_amp.square().mean().clamp_min(1e-8)
        loss = F.mse_loss(recon_amp * scale, target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        print(f"step {step + 1:03d}/{args.steps}: loss={losses[-1]:.6f}")

    model.eval()
    with torch.no_grad():
        final_phase = model(network_target)[..., :height, :width]
        recon = propagation_asm(torch.polar(torch.ones_like(final_phase), final_phase),
                                model.feature_size, wavelength, model.distance).abs()
        recon = (recon / recon.max().clamp_min(1e-8)).squeeze().cpu().numpy()
    save_phase(final_phase, os.path.join(args.output, "phase.png"))
    cv2.imwrite(os.path.join(args.output, "reconstruction.png"), np.uint8(np.clip(recon, 0, 1) * 255))
    torch.save(model.state_dict(), os.path.join(args.output, "holonet.pth"))
    with open(os.path.join(args.output, "loss.csv"), "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("step", "mse"))
        writer.writerows(enumerate(losses, start=1))
    print(f"Saved phase, reconstruction, weights, and loss log to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
