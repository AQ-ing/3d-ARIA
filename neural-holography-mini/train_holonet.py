import argparse
import json
import os
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from holonet import DepthControlNet, HoloNet
from propagation import propagation_asm
from utils.augmented_image_loader import ImageLoader
from utils.perceptualloss import PerceptualLoss


# ==================== 加载 HoloNet 权重 ====================
def load_holonet_checkpoint(model, path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("holonet", checkpoint.get("model", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise ValueError(f"Checkpoint does not match HoloNet (missing={missing}, unexpected={unexpected})")


# ==================== 振幅损失：逐图亮度匹配后计算感知损失 ====================
def amplitude_loss(reconstruction, target, loss_fn):
    scale = (reconstruction * target).mean(dim=(1, 2, 3), keepdim=True)
    scale = scale / reconstruction.square().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)
    return loss_fn(reconstruction * scale, target), scale


def merged_amplitude_loss(reconstructions, masks, target, loss_fn):
    """Select each plane's region before evaluating the complete image."""
    merged = sum(recon * mask for recon, mask in zip(reconstructions, masks))
    loss, scale = amplitude_loss(merged, target, loss_fn)
    return loss, scale, merged


def main():
    # ==================== 1. 解析训练参数 ====================
    parser = argparse.ArgumentParser(description="Train HoloNet on a directory of RGB/depth pairs.")
    parser.add_argument("--image", default='D:/wy/3d-ARIA/data/DIV2K_train_HR')
    parser.add_argument("--depth", default='D:/wy/3d-ARIA/data/DIV2K_train_HR_depth')
    parser.add_argument("--channel", type=int, choices=(0, 1, 2), default=1, help="RGB channel: 0, 1, or 2")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--size", type=int, nargs=2, default=(1920, 1080), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--controlnet-mode", choices=("off", "joint", "controlnet-only"), default="off")
    parser.add_argument("--init-checkpoint", help="HoloNet weights, required for controlnet-only mode")
    parser.add_argument("--distance-near", type=float, default=0.18, help="Near-plane propagation distance in meters")
    parser.add_argument("--distance-far", type=float, default=0.22, help="Far-plane propagation distance in meters")
    parser.add_argument("--depth-threshold", type=float, default=0.1, help="Normalized depth threshold separating the two planes")
    parser.add_argument("--near-is", choices=("dark", "bright"), default="bright")
    parser.add_argument("--runid", default=datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    parser.add_argument("--checkpoint-dir", default="checkpoint")
    args = parser.parse_args()

    # ==================== 2. 检查参数有效性 ====================
    if not args.runid or any(not (c.isalnum() or c in "_-") for c in args.runid):
        parser.error("--runid may contain only letters, numbers, underscores and hyphens")
    width, height = args.size
    if min(args.epochs, args.batch_size) < 1 or min(width, height) < 16:
        parser.error("--epochs and --batch-size must be positive and both dimensions must be at least 16")
    if not 0.0 < args.depth_threshold < 1.0:
        parser.error("--depth-threshold must be between 0 and 1")
    if args.controlnet_mode != "off" and args.distance_near == args.distance_far:
        parser.error("ControlNet mode requires two distinct propagation distances")
    if args.controlnet_mode != "off" and min(args.distance_near, args.distance_far) <= 0:
        parser.error("ControlNet propagation distances must be positive")
    if args.controlnet_mode == "controlnet-only" and not args.init_checkpoint:
        parser.error("--init-checkpoint is required for controlnet-only mode")

    # ==================== 3. 设置设备、波长与数据 loader ====================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wavelength = (638e-9, 520e-9, 450e-9)[args.channel]
    image_loader = ImageLoader(args.image, channel=args.channel, batch_size=args.batch_size,
                               image_res=(height, width), shuffle=True,
                               vertical_flips=False, horizontal_flips=False, direct_resize=True,
                               depth_path=args.depth if args.controlnet_mode != "off" else None)
    num_batches = (len(image_loader.im_names) + args.batch_size - 1) // args.batch_size
    padded_height = (height + 15) // 16 * 16
    padded_width = (width + 15) // 16 * 16

    # ==================== 4. 初始化 HoloNet 并加载已有权重 ====================
    model_distance = (args.distance_near + args.distance_far) / 2 if args.controlnet_mode != "off" else 0.2
    model = HoloNet(wavelength=wavelength, distance=model_distance).to(device)
    if args.init_checkpoint:
        load_holonet_checkpoint(model, args.init_checkpoint, device)

    # ==================== 5. 创建 ControlNet，设置双深度 ====================
    controlnet = None
    if args.controlnet_mode != "off":
        controlnet = DepthControlNet(model.final_phase_only.net[0]).to(device)
        distances = (args.distance_near, args.distance_far)

    # ==================== 6. 选择可训练参数，创建优化器与感知损失 ====================
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
    # 与原项目默认 vgg-low 一致：像素项 + 0.025 × VGG 特征项。
    loss_fn = PerceptualLoss(lambda_feat=0.025).to(device).eval()

    # ==================== 7. 按 runid 保存配置，拒绝覆盖已有运行 ====================
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    config = vars(args).copy()
    config.update({"device": str(device), "wavelength": wavelength, "base_distance": model.distance})
    config["loss_type"] = "merged_amplitude_perceptual" if controlnet is not None else "single_plane_amplitude_perceptual"
    config["perceptual_lambda_feat"] = loss_fn.lambda_feat
    config_path = os.path.join(args.checkpoint_dir, f"{args.runid}_config.json")
    with open(config_path, "x", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
    suffix = args.controlnet_mode.replace("-", "_")
    # ==================== 8. TensorBoard 按 runid 记录训练 ====================
    log_dir = os.path.join(args.checkpoint_dir, "tensorboard", args.runid)
    with SummaryWriter(log_dir=log_dir) as writer:
        for epoch in range(args.epochs):
            epoch_loss, num_images = 0.0, 0
            for batch, loaded in enumerate(image_loader):
                target = loaded[0]
                depth = loaded[3] if controlnet is not None else None
                # 8.1 准备当前 batch：振幅、深度分区与网络输入补齐
                target = target.to(device)
                network_target = F.pad(target, (0, padded_width - width, 0, padded_height - height), mode="replicate")
                if controlnet is not None:
                    depth = depth.to(device)
                    near_mask = depth <= args.depth_threshold
                    if args.near_is == "bright":
                        near_mask = ~near_mask
                    masks = (near_mask.float(), (~near_mask).float())
                    depth = F.pad(depth, (0, padded_width - width, 0, padded_height - height), mode="replicate")
                # 8.2 前向生成相位，构造 SLM 纯相位光场
                optimizer.zero_grad(set_to_none=True)
                phase = model(network_target, depth=depth if controlnet is not None else None, controlnet=controlnet)
                phase = phase[..., :height, :width]
                slm_field = torch.polar(torch.ones_like(phase), phase)

                # 8.3 ASM 传播与感知损失：双深度先按掩膜拼接，再比较完整目标
                if controlnet is None:
                    recon_amp = propagation_asm(slm_field, model.feature_size, wavelength, model.distance).abs()
                    loss, _ = amplitude_loss(recon_amp, target, loss_fn)
                else:
                    reconstructions = [propagation_asm(slm_field, model.feature_size, wavelength, distance).abs()
                                       for distance in distances]
                    loss, _, _ = merged_amplitude_loss(reconstructions, masks, target, loss_fn)

                # 8.4 反向传播与参数更新
                loss.backward()
                optimizer.step()
                # 8.5 记录当前 batch 的损失
                loss_value = loss.item()
                global_step = epoch * num_batches + batch + 1
                writer.add_scalar("Loss/train_step", loss_value, global_step)
                epoch_loss += loss_value * target.shape[0]
                num_images += target.shape[0]
                print(f"epoch {epoch + 1:02d}/{args.epochs} batch {batch + 1}/{num_batches}: perceptual_loss={loss_value:.6f}")

            # ==================== 9. 每个 epoch 保存一次权重 ====================
            tag = f"{args.runid}_{suffix}_epoch_{epoch + 1:03d}"
            checkpoint = {"holonet": model.state_dict(), "config": config,
                          "epoch": epoch + 1, "global_step": global_step}
            if controlnet is not None:
                checkpoint["controlnet"] = controlnet.state_dict()
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, f"{tag}.pth"))

            # ==================== 10. 记录逐图平均的 epoch 损失并写盘 ====================
            writer.add_scalar("Loss/train_epoch", epoch_loss / num_images, epoch + 1)
            writer.add_scalar("Learning_rate", optimizer.param_groups[0]["lr"], epoch + 1)
            writer.flush()
            print(f"Saved epoch {epoch + 1} weights: {tag}")


# ==================== 程序入口 ====================
if __name__ == "__main__":
    main()
