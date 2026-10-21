import argparse
import csv
import json
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from holonet import DepthControlNet, HoloNet
from propagation import propagation_asm


def load_rgb_target(path, channel, height, width, device):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read RGB image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image[..., channel].astype(np.float32) / 255.0
    image = np.where(image <= 0.04045, image / 12.92, ((image + 0.055) / 1.055) ** 2.4)
    tensor = torch.from_numpy(np.sqrt(image))[None, None]
    tensor = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
    return tensor.to(device)


def load_depth(path, height, width, device):
    depth = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if depth is None:
        raise FileNotFoundError(f"Could not read depth image: {path}")
    depth = torch.from_numpy(depth.astype(np.float32) / 255.0)[None, None]
    depth = F.interpolate(depth, size=(height, width), mode="bilinear", align_corners=False)
    return depth.to(device)


def load_holonet_checkpoint(model, path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("holonet", checkpoint.get("model", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise ValueError(f"Checkpoint does not match HoloNet (missing={missing}, unexpected={unexpected})")


def save_phase(phase, path):
    phase = phase.detach().squeeze().cpu().numpy()
    phase_u8 = np.round((1.0 - ((phase + np.pi) % (2 * np.pi)) / (2 * np.pi)) * 255).astype(np.uint8)
    if not cv2.imwrite(path, phase_u8):
        raise OSError(f"Could not write image: {path}")


def save_amplitude(amplitude, path):
    amplitude = amplitude.detach().squeeze().cpu().numpy()
    image = np.uint8(np.clip(amplitude, 0, 1) * 255)
    if not cv2.imwrite(path, image):
        raise OSError(f"Could not write image: {path}")


def merged_amplitude_loss(reconstructions, masks, target):
    """Select each plane's region before fitting one global amplitude scale."""
    merged = sum(recon * mask for recon, mask in zip(reconstructions, masks))
    scale = (merged * target).mean() / merged.square().mean().clamp_min(1e-8)
    loss = F.mse_loss(merged * scale, target)
    return loss, scale, merged


def main():
    parser = argparse.ArgumentParser(description="Overfit HoloNet to a single RGB/depth pair.")
    parser.add_argument("--image", default="data/image/chinese.png")
    parser.add_argument("--depth", default="data/depth/chinese.png")
    parser.add_argument("--channel", type=int, choices=(0, 1, 2), default=1, help="RGB channel: 0, 1, or 2")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--size", type=int, nargs=2, default=(1920, 1080), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--controlnet-mode", choices=("off", "joint", "controlnet-only"), default="off")
    parser.add_argument("--init-checkpoint", help="HoloNet weights, required for controlnet-only mode")
    parser.add_argument("--distance-near", type=float, default=0.18, help="Near-plane propagation distance in meters")
    parser.add_argument("--distance-far", type=float, default=0.22, help="Far-plane propagation distance in meters")
    parser.add_argument("--depth-threshold", type=float, default=0.1, help="Normalized depth threshold separating the two planes")
    parser.add_argument("--near-is", choices=("dark", "bright"), default="bright")
    parser.add_argument("--output", default="output")
    parser.add_argument("--checkpoint-dir", default="checkpoint")
    args = parser.parse_args()

    width, height = args.size
    if args.steps < 1 or min(width, height) < 16:
        parser.error("--steps must be positive and both output dimensions must be at least 16")
    if not 0.0 < args.depth_threshold < 1.0:
        parser.error("--depth-threshold must be between 0 and 1")
    if args.controlnet_mode != "off" and args.distance_near == args.distance_far:
        parser.error("ControlNet mode requires two distinct propagation distances")
    if args.controlnet_mode != "off" and min(args.distance_near, args.distance_far) <= 0:
        parser.error("ControlNet propagation distances must be positive")
    if args.controlnet_mode == "controlnet-only" and not args.init_checkpoint:
        parser.error("--init-checkpoint is required for controlnet-only mode")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wavelength = (638e-9, 520e-9, 450e-9)[args.channel]
    target = load_rgb_target(args.image, args.channel, height, width, device)
    padded_height = (height + 15) // 16 * 16
    padded_width = (width + 15) // 16 * 16
    network_target = F.pad(target, (0, padded_width - width, 0, padded_height - height), mode="replicate")

    model_distance = (args.distance_near + args.distance_far) / 2 if args.controlnet_mode != "off" else 0.2
    model = HoloNet(wavelength=wavelength, distance=model_distance).to(device)
    if args.init_checkpoint:
        load_holonet_checkpoint(model, args.init_checkpoint, device)

    controlnet = None
    if args.controlnet_mode != "off":
        controlnet = DepthControlNet(model.final_phase_only.net[0]).to(device)
        depth = load_depth(args.depth, height, width, device)
        depth = F.pad(depth, (0, padded_width - width, 0, padded_height - height), mode="replicate")
        near_mask = depth[..., :height, :width] <= args.depth_threshold
        if args.near_is == "bright":
            near_mask = ~near_mask
        far_mask = ~near_mask
        if not near_mask.any() or not far_mask.any():
            parser.error("The depth threshold must split the depth map into two non-empty regions")
        masks = (near_mask.float(), far_mask.float())
        plane_targets = tuple(target * mask for mask in masks)
        distances = (args.distance_near, args.distance_far)

    if args.controlnet_mode == "controlnet-only":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        trainable = controlnet.parameters()
    elif args.controlnet_mode == "joint":
        trainable = list(model.parameters()) + list(controlnet.parameters())
    else:
        trainable = model.parameters()
    optimizer = torch.optim.Adam(trainable, lr=args.learning_rate)

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    losses = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        phase = model(network_target, depth=depth if controlnet is not None else None, controlnet=controlnet)
        phase = phase[..., :height, :width]
        slm_field = torch.polar(torch.ones_like(phase), phase)

        if controlnet is None:
            recon_amp = propagation_asm(slm_field, model.feature_size, wavelength, model.distance).abs()
            scale = (recon_amp * target).mean() / recon_amp.square().mean().clamp_min(1e-8)
            loss = F.mse_loss(recon_amp * scale, target)
        else:
            reconstructions = [propagation_asm(slm_field, model.feature_size, wavelength, distance).abs()
                               for distance in distances]
            loss, scale, merged = merged_amplitude_loss(reconstructions, masks, target)
            # Diagnostics: area-weighted contributions add up to the merged MSE.
            with torch.no_grad():
                squared_error = (merged * scale - target).square()
                plane_losses = [(squared_error * mask).mean() for mask in masks]

        loss.backward()
        optimizer.step()
        if controlnet is None:
            losses.append((step + 1, float(loss.detach())))
            print(f"step {step + 1:03d}/{args.steps}: loss={losses[-1][-1]:.6f}")
        else:
            loss_values = [float(item.detach()) for item in plane_losses]
            losses.append((step + 1, *loss_values, float(loss.detach())))
            print(f"step {step + 1:03d}/{args.steps}: near_contribution={loss_values[0]:.6f} "
                  f"far_contribution={loss_values[1]:.6f} merged_mse={losses[-1][-1]:.6f}")

    model.eval()
    if controlnet is not None:
        controlnet.eval()
    with torch.no_grad():
        final_phase = model(network_target, depth=depth if controlnet is not None else None, controlnet=controlnet)
        final_phase = final_phase[..., :height, :width]
        save_phase(final_phase, os.path.join(args.output, "phase.png"))
        if controlnet is None:
            recon = propagation_asm(torch.polar(torch.ones_like(final_phase), final_phase),
                                    model.feature_size, wavelength, model.distance).abs()
            recon = recon / recon.max().clamp_min(1e-8)
            save_amplitude(recon, os.path.join(args.output, "reconstruction.png"))
        else:
            slm_field = torch.polar(torch.ones_like(final_phase), final_phase)
            reconstructions = [propagation_asm(slm_field, model.feature_size, wavelength, distance).abs()
                               for distance in distances]
            _, scale, merged = merged_amplitude_loss(reconstructions, masks, target)
            save_amplitude(merged * scale, os.path.join(args.output, "reconstruction_merged.png"))
            save_amplitude(target, os.path.join(args.output, "target.png"))
            for index, (recon, goal) in enumerate(zip(reconstructions, plane_targets), start=1):
                recon = recon / recon.max().clamp_min(1e-8)
                save_amplitude(recon, os.path.join(args.output, f"reconstruction_plane{index}.png"))
                save_amplitude(goal, os.path.join(args.output, f"target_plane{index}.png"))

    suffix = args.controlnet_mode.replace("-", "_")
    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"holonet_{suffix}.pth"))
    if controlnet is not None:
        torch.save(controlnet.state_dict(), os.path.join(args.checkpoint_dir, f"controlnet_{suffix}.pth"))
    loss_path = os.path.join(args.checkpoint_dir, f"loss_{suffix}.csv")
    with open(loss_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("step", "near_mse_contribution", "far_mse_contribution", "merged_mse") if controlnet is not None
                        else ("step", "mse"))
        writer.writerows(losses)
    config = vars(args).copy()
    config.update({"device": str(device), "wavelength": wavelength, "base_distance": model.distance})
    config["loss_type"] = "merged_amplitude_mse" if controlnet is not None else "single_plane_amplitude_mse"
    with open(os.path.join(args.checkpoint_dir, f"config_{suffix}.json"), "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
    print(f"Saved output images to: {os.path.abspath(args.output)}")
    print(f"Saved weights and training records to: {os.path.abspath(args.checkpoint_dir)}")


if __name__ == "__main__":
    main()
