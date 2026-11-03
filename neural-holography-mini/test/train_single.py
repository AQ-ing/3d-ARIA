"""Overfit one image with the current HoloNet/ControlNet training loss."""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from holonet import DepthControlNet, HoloNet
from propagation import propagation_asm
from train_holonet import amplitude_loss, load_holonet_checkpoint, merged_amplitude_loss
from utils.augmented_image_loader import ImageLoader, pad_crop_to_res, resize_keep_aspect
from utils.perceptualloss import PerceptualLoss
from utils.utils import crop_image


def main():
    global HoloNet, propagation_asm
    # ==================== 1. 单图训练参数 ====================
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="One RGB image")
    parser.add_argument("--off-reference", action="store_true", help="Original ASM and fixed 0.95 intensity training scale; off only")
    parser.add_argument("--depth", type=Path, help="Matching depth image for ControlNet modes")
    parser.add_argument("--controlnet-mode", choices=("off", "joint", "controlnet-only"), default="off")
    parser.add_argument("--init-checkpoint", type=Path, help="Pretrained HoloNet checkpoint for controlnet-only")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--size", type=int, nargs=2, default=(1920, 1080), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--preprocess", choices=("direct", "homography"), default="direct")
    parser.add_argument("--loss-region", choices=("full", "roi"), default="roi")
    parser.add_argument("--loss", choices=("vgg-low", "mse"), default="vgg-low")
    parser.add_argument("--channel", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--distance-near", type=float, default=0.18)
    parser.add_argument("--distance-far", type=float, default=0.22)
    parser.add_argument("--same-plane-diagnostic", action="store_true", help="ControlNet diagnostic only: both planes at 0.2 m")
    parser.add_argument("--depth-threshold", type=float, default=0.1)
    parser.add_argument("--near-is", choices=("dark", "bright"), default="bright")
    parser.add_argument("--output", type=Path, help="Output directory inside test/ (default: test/output/<mode>)")
    parser.add_argument("--save-model", type=Path, help="Optional final HoloNet weights inside test/")
    args = parser.parse_args()

    use_depth = args.controlnet_mode != "off"
    if args.same_plane_diagnostic:
        if not use_depth:
            parser.error("--same-plane-diagnostic requires ControlNet")
        args.distance_near = args.distance_far = 0.2
    if args.off_reference:
        if use_depth:
            parser.error("--off-reference only supports off")
        from reference_off import ReferenceHoloNet, propagation_asm as reference_asm
        HoloNet, propagation_asm = ReferenceHoloNet, reference_asm
    if not args.image.is_file() or (use_depth and (args.depth is None or not args.depth.is_file())):
        parser.error("--image must be a file; ControlNet modes also require a depth file")
    if args.controlnet_mode == "controlnet-only" and not args.init_checkpoint:
        parser.error("--init-checkpoint is required for controlnet-only")
    width, height = args.size
    if min(args.steps, args.log_every) < 1 or min(width, height) < 16 or args.learning_rate <= 0:
        parser.error("--steps, --log-every and --learning-rate must be positive; image dimensions must be >= 16")
    if use_depth and (not 0 < args.depth_threshold < 1 or
                      min(args.distance_near, args.distance_far) <= 0 or
                      (args.distance_near == args.distance_far and not args.same_plane_diagnostic)):
        parser.error("ControlNet requires two positive distinct distances and a threshold in (0, 1)")
    output_dir = (args.output or Path(__file__).resolve().parent / "output" / args.controlnet_mode).resolve()
    if not output_dir.is_relative_to(Path(__file__).resolve().parent):
        parser.error("--output must be inside the test directory")
    if args.save_model and not args.save_model.resolve().is_relative_to(Path(__file__).resolve().parent):
        parser.error("--save-model must be inside the test directory")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ==================== 2. 读取同一张图片与深度 ====================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wavelength = (638e-9, 520e-9, 450e-9)[args.channel]
    roi = (round(width * 880 / 1920), round(width * 1600 / 1920)) if args.preprocess == "homography" else (height, width)
    if roi[0] > height:
        parser.error("--size height is too small for the homography region")
    loader = ImageLoader(str(args.image), channel=args.channel, batch_size=1,
                         image_res=(height, width), homography_res=roi, shuffle=False,
                         vertical_flips=False, horizontal_flips=False,
                         direct_resize=args.preprocess == "direct",
                         depth_path=str(args.depth) if use_depth and args.preprocess == "direct" else None)
    loaded = next(iter(loader))
    target = loaded[0].to(device)
    depth = None
    if use_depth:
        if args.preprocess == "direct":
            depth = loaded[3].to(device)
        else:
            depth_image = cv2.imread(str(args.depth), cv2.IMREAD_GRAYSCALE)
            if depth_image is None or depth_image.shape != loaded[1][0]:
                parser.error("Depth must be readable and match the RGB image dimensions")
            depth_image = resize_keep_aspect(depth_image[None].astype(np.float32) / 255, roi)
            depth_image = pad_crop_to_res(depth_image, (height, width))
            depth = torch.from_numpy(np.ascontiguousarray(depth_image))[None].to(device)
    padding = (0, (-width) % 16, 0, (-height) % 16)
    network_target = F.pad(target, padding, mode="replicate")
    network_depth = F.pad(depth, padding, mode="replicate") if use_depth else None
    if use_depth:
        loss_depth = crop_image(depth, roi, stacked_complex=False) if args.loss_region == "roi" else depth
        near = loss_depth > args.depth_threshold if args.near_is == "bright" else loss_depth <= args.depth_threshold
        masks = (near.float(), (~near).float())
        distances = (args.distance_near, args.distance_far)
    loss_target = crop_image(target, roi, stacked_complex=False) if args.loss_region == "roi" else target

    # ==================== 3. 构建网络与当前版本的感知损失 ====================
    distance = sum(distances) / 2 if use_depth else 0.2
    model = HoloNet(wavelength=wavelength, distance=distance).to(device)
    if args.init_checkpoint:
        load_holonet_checkpoint(model, args.init_checkpoint, device)
    controlnet = DepthControlNet(model.final_phase_only.net[0]).to(device) if use_depth else None
    if args.controlnet_mode == "controlnet-only":
        model.requires_grad_(False).eval()
        parameters = controlnet.parameters()
    elif use_depth:
        parameters = list(model.parameters()) + list(controlnet.parameters())
    else:
        parameters = model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
    loss_fn = PerceptualLoss(lambda_feat=0.025).to(device).eval() if args.loss == "vgg-low" else F.mse_loss
    forward_kernel = propagation_asm(target, model.feature_size, wavelength, distance, return_kernel=True) if args.off_reference else None

    # ==================== 4. 重复优化同一张图片 ====================
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        phase = model(network_target, depth=network_depth, controlnet=controlnet)[..., :height, :width]
        slm_field = torch.polar(torch.ones_like(phase), phase)
        if use_depth:
            reconstructions = [propagation_asm(slm_field, model.feature_size, wavelength, z).abs()
                               for z in distances]
            if args.loss_region == "roi":
                reconstructions = [crop_image(recon, roi, stacked_complex=False) for recon in reconstructions]
            loss, _, _ = merged_amplitude_loss(reconstructions, masks, loss_target, loss_fn)
        else:
            reconstruction = propagation_asm(slm_field, model.feature_size, wavelength, distance, kernel=forward_kernel).abs()
            if args.loss_region == "roi":
                reconstruction = crop_image(reconstruction, roi, stacked_complex=False)
            if args.off_reference:
                loss = loss_fn(reconstruction * np.sqrt(0.95), loss_target)
            else:
                loss, _ = amplitude_loss(reconstruction, loss_target, loss_fn)
        loss.backward()
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(f"step {step}/{args.steps}: {args.loss}_loss={loss.item():.6f}", flush=True)

    # ==================== 5. 训练完成，仅输出最终重建图 ====================
    model.eval()
    if controlnet is not None:
        controlnet.eval()
    with torch.inference_mode():
        phase = model(network_target, depth=network_depth, controlnet=controlnet)[..., :height, :width]
        slm_field = torch.polar(torch.ones_like(phase), phase)
        if use_depth:
            reconstructions = [propagation_asm(slm_field, model.feature_size, wavelength, z).abs()
                               for z in distances]
            if args.preprocess == "homography":
                reconstructions = [crop_image(recon, roi, stacked_complex=False) for recon in reconstructions]
                output_depth = crop_image(depth, roi, stacked_complex=False)
                output_near = output_depth > args.depth_threshold if args.near_is == "bright" else output_depth <= args.depth_threshold
                output_masks = (output_near.float(), (~output_near).float())
            else:
                output_masks = masks
            reconstruction = sum(recon * mask for recon, mask in zip(reconstructions, output_masks))
            name = f"mergerecon-{args.image.stem}.png"
        else:
            reconstruction = propagation_asm(slm_field, model.feature_size, wavelength, distance).abs()
            if args.preprocess == "homography":
                reconstruction = crop_image(reconstruction, roi, stacked_complex=False)
            name = f"recon-{args.image.stem}.png"
        output_target = crop_image(target, roi, stacked_complex=False) if args.preprocess == "homography" else target
        final_loss, scale = amplitude_loss(reconstruction, output_target, loss_fn)
        mse = F.mse_loss(reconstruction * scale, output_target).item()
        if args.off_reference:
            fixed_mse = F.mse_loss(reconstruction * np.sqrt(0.95), output_target).item()
            print(f"reference_fixed_scale_psnr={-10 * np.log10(max(fixed_mse, 1e-12)):.6f} dB", flush=True)
        psnr = -10 * np.log10(max(mse, 1e-12))
        if use_depth:
            squared_error = (reconstruction * scale - output_target).square()
            for label, mask in zip(("near", "far"), output_masks):
                region_mse = (squared_error * mask).sum() / mask.sum().clamp_min(1)
                region_psnr = -10 * np.log10(max(region_mse.item(), 1e-12))
                print(f"{label}: pixels={mask.mean().item():.1%} mse={region_mse.item():.6f} psnr={region_psnr:.2f} dB", flush=True)
        pixels = np.uint8(np.clip((reconstruction * scale).squeeze().cpu().numpy(), 0, 1) * 255)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / name
        if not cv2.imwrite(str(output_path), pixels):
            raise OSError(f"Could not save {output_path}")
        if args.save_model:
            args.save_model.parent.mkdir(parents=True, exist_ok=True)
            weights = {"holonet": model.state_dict()}
            if controlnet is not None:
                weights["controlnet"] = controlnet.state_dict()
            torch.save(weights, args.save_model)
        print(f"final {args.loss}_loss={final_loss.item():.6f} amplitude_mse={mse:.9f} amplitude_psnr={psnr:.6f} dB", flush=True)
        print(f"Saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
