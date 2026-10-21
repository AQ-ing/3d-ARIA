# HoloNet with two-plane depth ControlNet

This mini project overfits one RGB/depth image pair. It supports the original single-plane HoloNet and a two-plane depth-conditioned ControlNet.

## Data

RGB and depth files are paired by filename under `data/image/` and `data/depth/`. The provided pairs are `chinese.png` and `drink.png`. The depth map is resized with the RGB image and normalized to `[0, 1]`; values are split at `--depth-threshold`. By default, dark values target the near plane and bright values target the far plane. Use `--near-is bright` to reverse that mapping.

The phase hologram is evaluated at two propagation distances in ControlNet modes. Its targets are the RGB image masked by the two depth regions. The combined multi-plane loss supervises both the desired image region and suppression of that region at the other plane.

## Modes

- `off`: original HoloNet single-plane training; no depth image is loaded.
- `joint`: train HoloNet and the copied ControlNet together from scratch, or from `--init-checkpoint` if supplied.
- `controlnet-only`: load a completed HoloNet with `--init-checkpoint`, freeze it, and train only the copied ControlNet branch.

The ControlNet copies the final-phase U-Net. A zero-initialized convolution adds the depth condition to the copied U-Net input, and zero-initialized 1x1 convolutions inject its multiscale features into the base U-Net.

## Examples

Run the baseline:

```powershell
conda run -n e2e python train_holonet.py --controlnet-mode off --image data/image/chinese.png --steps 100
```

Jointly train for two depth planes at 0.18 m and 0.22 m:

```powershell
conda run -n e2e python train_holonet.py --controlnet-mode joint --image data/image/chinese.png --depth data/depth/chinese.png --steps 100
```

Freeze a trained HoloNet and train only ControlNet:

```powershell
conda run -n e2e python train_holonet.py --controlnet-mode controlnet-only --init-checkpoint checkpoint/holonet_off.pth --image data/image/chinese.png --depth data/depth/chinese.png --steps 100
```

`--size` accepts width and height and defaults to `512 384`. The network pads dimensions to a multiple of 16 and crops the generated phase back to the requested size. `--distance-near`, `--distance-far`, `--depth-threshold`, and `--near-is` configure the two-plane mapping.

Generated images only go to `output/`. Weights, loss CSV files, and run configuration JSON files go to `checkpoint/`.

## Preview the depth threshold without training

```powershell
conda run -n e2e python train_holonet.py --preview-only --image data/image/chinese.png --depth data/depth/chinese.png --depth-threshold 0.5 --near-is dark --output output/preview_chinese
```

The threshold is a normalized grayscale value, not a distance or a percentage of the image area: `0.5` corresponds to grayscale `127.5/255`. With `--near-is dark`, plane 1 is `D <= threshold` and plane 2 is `D > threshold`. `--near-is bright` swaps these two assignments. This orientation is an explicit choice; relative/inverse-depth maps often use brighter values for nearer objects.

The preview saves `depth_partition_preview.png` (RGB, depth, masks and RGB regions), binary `mask_plane1.png` / `mask_plane2.png`, color `rgb_plane1.png` / `rgb_plane2.png`, and the actual single-channel amplitude `target_plane1.png` / `target_plane2.png`. Preview and training share the exact same masks, computed after bilinear resizing of the normalized depth image. ControlNet continues to receive the continuous depth map; these binary masks define the two training targets. The preview-only command creates no model or checkpoint. ControlNet training also saves the preview before the first optimization step.
