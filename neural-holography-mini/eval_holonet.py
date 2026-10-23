"""Run HoloNet inference and save recon or mergerecon images."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from holonet import DepthControlNet, HoloNet
from propagation import propagation_asm
from utils.augmented_image_loader import ImageLoader


# ==================== 图片保存工具 ====================
def save_amplitude(amplitude, path):
    pixels = amplitude.squeeze().cpu().numpy()
    if not cv2.imwrite(str(path), np.uint8(np.clip(pixels, 0, 1) * 255)):
        raise OSError(f"Could not save image: {path}")


def main():
    # ==================== 1. 指定推理模式、权重、输入与输出 ====================
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controlnet-mode", choices = ("off", "joint", "controlnet-only"), required=True)
    parser.add_argument("--checkpoint", required = True, help="Trained checkpoint containing model weights and inference settings")
    parser.add_argument("--image", default = '', required=True, help="RGB image file or directory")
    parser.add_argument("--depth", default = '', help="Paired depth file or directory; required when ControlNet is enabled")
    parser.add_argument("--output", default = "output")
    args = parser.parse_args()
    use_depth = args.controlnet_mode != "off"
    if use_depth and not args.depth:
        parser.error("--depth is required when ControlNet is enabled")

    # ==================== 2. 从 checkpoint 加载模型与推理参数 ====================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = checkpoint.get("config", {})
    width, height = config.get("size", (1920, 1080))
    channel = config.get("channel", 1)
    wavelength = (638e-9, 520e-9, 450e-9)[channel]
    distances = (config.get("distance_near", 0.18), config.get("distance_far", 0.22))
    distance = config.get("base_distance", sum(distances) / 2 if use_depth else 0.2)
    model = HoloNet(wavelength=wavelength, distance=distance).to(device)
    model.load_state_dict(checkpoint.get("holonet", checkpoint.get("model", checkpoint)))
    model.eval()
    controlnet = None
    if use_depth:
        control_state = checkpoint.get("controlnet")
        # 兼容之前分别保存的同名 HoloNet / ControlNet 权重。
        if control_state is None and checkpoint_path.name.endswith("_holonet.pth"):
            control_path = checkpoint_path.with_name(checkpoint_path.name[:-len("_holonet.pth")] + "_controlnet.pth")
            if control_path.is_file():
                control_state = torch.load(control_path, map_location=device, weights_only=True)
        if control_state is None:
            parser.error("The selected mode requires trained ControlNet weights in the checkpoint")
        controlnet = DepthControlNet(model.final_phase_only.net[0]).to(device)
        controlnet.load_state_dict(control_state)
        controlnet.eval()

    # ==================== 3. 读取输入图片及配对深度 ====================
    loader = ImageLoader(args.image, channel=channel, batch_size=1,
                         image_res=(height, width), shuffle=False,
                         vertical_flips=False, horizontal_flips=False, direct_resize=True,
                         depth_path=args.depth if use_depth else None)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    padding = (0, (-width) % 16, 0, (-height) % 16)
    with torch.inference_mode():
        for loaded in loader:
            amplitude, _, names = loaded[:3]
            amplitude = amplitude.to(device)
            depth = loaded[3].to(device) if use_depth else None

            # ==================== 4. 前向生成相位，ASM 传播生成重建图 ====================
            phase = model(F.pad(amplitude, padding, mode="replicate"),
                          depth=F.pad(depth, padding, mode="replicate") if use_depth else None,
                          controlnet=controlnet)[..., :height, :width]
            field = torch.polar(torch.ones_like(phase), phase)
            planes = distances if use_depth else (distance,)
            reconstructions = [propagation_asm(field, model.feature_size, wavelength, z).abs() for z in planes]
            # 仅为图片显示归一化；双平面共用比例，保留相对亮度。
            display_scale = torch.stack([recon.amax() for recon in reconstructions]).amax().clamp_min(1e-8)

            # ==================== 5. 将推理图片保存到指定目录 ====================
            image_name = Path(names[0]).name
            if use_depth:
                near = depth <= config.get("depth_threshold", 0.1)
                if config.get("near_is", "bright") == "bright":
                    near = ~near
                masks = (near.float(), (~near).float())
                merged = sum(recon * mask for recon, mask in zip(reconstructions, masks))
                output_path = output / f"mergerecon-{image_name}.png"
                save_amplitude(merged / display_scale, output_path)
            else:
                output_path = output / f"recon-{image_name}.png"
                save_amplitude(reconstructions[0] / display_scale, output_path)
            print(f"Saved inference image to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
