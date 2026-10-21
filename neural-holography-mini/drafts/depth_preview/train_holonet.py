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


def split_depth(depth, threshold, near_is):
    """The same threshold masks are used for training targets and previews."""
    near_mask = depth <= threshold
    if near_is == "bright":
        near_mask = ~near_mask
    return near_mask.float(), (~near_mask).float()


def save_depth_preview(image_path, depth, masks, threshold, near_is, distances, output):
    height, width = depth.shape[-2:]
    rgb = cv2.imread(image_path, cv2.IMREAD_COLOR)
    rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    gray = np.uint8(np.clip(depth.squeeze().cpu().numpy(), 0, 1) * 255)
    mask_images = [np.uint8(mask.squeeze().cpu().numpy() * 255) for mask in masks]
    images = [rgb, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)]
    titles = ["RGB input", f"Depth / 255; threshold = {threshold:g}"]
    for index, (mask, distance) in enumerate(zip(mask_images, distances), start=1):
        selected = cv2.bitwise_and(rgb, rgb, mask=mask)
        images.extend([cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), selected])
        relation = "<=" if (index == 1) == (near_is == "dark") else ">"
        titles.extend([f"Plane {index}: D {relation} {threshold:g} ({(mask > 0).mean():.1%})",
                       f"Plane {index} RGB / z = {distance:g} m"])
        for name, image in [(f"mask_plane{index}.png", mask), (f"rgb_plane{index}.png", selected)]:
            if not cv2.imwrite(os.path.join(output, name), image):
                raise OSError(f"Could not write preview: {name}")
    tile_width = 600
    tile_height = round(height * tile_width / width)
    rows = []
    for start in range(0, len(images), 2):
        tiles = []
        for image, title in zip(images[start:start + 2], titles[start:start + 2]):
            tile = cv2.resize(image, (tile_width, tile_height), interpolation=cv2.INTER_NEAREST)
            tile = cv2.copyMakeBorder(tile, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
            cv2.putText(tile, title, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.57, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)
        rows.append(np.concatenate(tiles, axis=1))
    preview_path = os.path.join(output, "depth_partition_preview.png")
    if not cv2.imwrite(preview_path, np.concatenate(rows, axis=0)):
        raise OSError(f"Could not write preview: {preview_path}")
    print(f"Depth partition preview: {os.path.abspath(preview_path)}")


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


def main():
    parser = argparse.ArgumentParser(description="Overfit HoloNet to a single RGB/depth pair.")
    parser.add_argument("--image", default="data/image/chinese.png")
    parser.add_argument("--depth", default="data/depth/chinese.png")
    parser.add_argument("--channel", type=int, choices=(0, 1, 2), default=1, help="RGB channel: 0, 1, or 2")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--size", type=int, nargs=2, default=(1920, 1080), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--controlnet-mode", choices=("off", "joint", "controlnet-only"), default="off")
    parser.add_argument("--init-checkpoint", help="HoloNet weights, required for controlnet-only mode")
    parser.add_argument("--distance-near", type=float, default=0.18, help="Near-plane propagation distance in meters")
    parser.add_argument("--distance-far", type=float, default=0.22, help="Far-plane propagation distance in meters")
    parser.add_argument("--depth-threshold", type=float, default=0.5, help="Normalized depth threshold separating the two planes")
    parser.add_argument("--near-is", choices=("dark", "bright"), default="dark")
    parser.add_argument("--preview-only", action="store_true", help="Save depth masks, RGB regions, and amplitude targets without training")
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
    if args.controlnet_mode == "controlnet-only" and not args.init_checkpoint and not args.preview_only:
        parser.error("--init-checkpoint is required for controlnet-only mode")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wavelength = (638e-9, 520e-9, 450e-9)[args.channel]
    target = load_rgb_target(args.image, args.channel, height, width, device)
    padded_height = (height + 15) // 16 * 16
    padded_width = (width + 15) // 16 * 16
    network_target = F.pad(target, (0, padded_width - width, 0, padded_height - height), mode="replicate")

    os.makedirs(args.output, exist_ok=True)
    if args.controlnet_mode != "off" or args.preview_only:
        depth = load_depth(args.depth, height, width, device)
        masks = split_depth(depth, args.depth_threshold, args.near_is)
        if not all(mask.any() for mask in masks):
            parser.error("The depth threshold must split the depth map into two non-empty regions")
        plane_targets = tuple(target * mask for mask in masks)
        distances = (args.distance_near, args.distance_far)
        save_depth_preview(args.image, depth, masks, args.depth_threshold, args.near_is, distances, args.output)
        for index, goal in enumerate(plane_targets, start=1):
            save_amplitude(goal, os.path.join(args.output, f"target_plane{index}.png"))
        if args.preview_only:
            return
        depth = F.pad(depth, (0, padded_width - width, 0, padded_height - height), mode="replicate")

    model_distance = (args.distance_near + args.distance_far) / 2 if args.controlnet_mode != "off" else 0.2
    model = HoloNet(wavelength=wavelength, distance=model_distance).to(device)
    if args.init_checkpoint:
        load_holonet_checkpoint(model, args.init_checkpoint, device)

    controlnet = None
    if args.controlnet_mode != "off":
        controlnet = DepthControlNet(model.final_phase_only.net[0]).to(device)

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
            plane_losses = [loss]
        else:
            reconstructions = [propagation_asm(slm_field, model.feature_size, wavelength, distance).abs()
                               for distance in distances]
            numerator = sum((recon * goal).mean() for recon, goal in zip(reconstructions, plane_targets))
            denominator = sum(recon.square().mean() for recon in reconstructions).clamp_min(1e-8)
            scale = numerator / denominator
            plane_losses = [F.mse_loss(recon * scale, goal)
                            for recon, goal in zip(reconstructions, plane_targets)]
            loss = torch.stack(plane_losses).mean()

        loss.backward()
        optimizer.step()
        loss_values = [float(item.detach()) for item in plane_losses]
        losses.append((step + 1, *loss_values, float(loss.detach())))
        if controlnet is None:
            print(f"step {step + 1:03d}/{args.steps}: loss={losses[-1][-1]:.6f}")
        else:
            print(f"step {step + 1:03d}/{args.steps}: near={loss_values[0]:.6f} "
                  f"far={loss_values[1]:.6f} total={losses[-1][-1]:.6f}")

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
            for index, (distance, goal) in enumerate(zip(distances, plane_targets), start=1):
                recon = propagation_asm(slm_field, model.feature_size, wavelength, distance).abs()
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
        writer.writerow(("step", "near_mse", "far_mse", "total_mse") if controlnet is not None
                        else ("step", "mse"))
        writer.writerows(losses)
    config = vars(args).copy()
    config.update({"device": str(device), "wavelength": wavelength, "base_distance": model.distance})
    with open(os.path.join(args.checkpoint_dir, f"config_{suffix}.json"), "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
    print(f"Saved output images to: {os.path.abspath(args.output)}")
    print(f"Saved weights and training records to: {os.path.abspath(args.checkpoint_dir)}")


if __name__ == "__main__":
    main()
