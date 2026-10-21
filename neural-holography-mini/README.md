# HoloNet with two-plane depth ControlNet

This mini project overfits one RGB/depth image pair. It supports the original single-plane HoloNet and a two-plane depth-conditioned ControlNet.

## Data

RGB and depth files are paired by filename under `data/image/` and `data/depth/`. The provided pairs are `chinese.png` and `drink.png`. The depth map is resized with the RGB image and normalized to `[0, 1]`; values are split at `--depth-threshold` (default `0.1`). By default, bright values target the near plane and dark values target the far plane. Use `--near-is dark` to reverse that mapping.

The phase hologram is evaluated at two propagation distances in ControlNet modes. With complementary masks `M_near` and `M_far`, the reconstructed amplitudes are merged as `R = M_near * R_near + M_far * R_far`. Training minimizes `mean((s * R - A)^2)` against the full target amplitude `A`, where `s = mean(R * A) / max(mean(R^2), 1e-8)`. The global scale is fitted after merging; values outside each plane's assigned region do not enter the loss. There is no additional suppression or region-balancing term. Single-plane mode retains its original scaled amplitude MSE.

For ControlNet runs, `reconstruction_merged.png` shows the final merged amplitude using the same scale as the loss (clipped to `[0, 1]` for PNG), and `target.png` shows the full target amplitude. Individual plane reconstructions remain normalized separately for display, so they should not be used to reproduce the merged loss. CSV columns `near_mse_contribution` and `far_mse_contribution` are area-weighted contributions whose sum equals `merged_mse`. Configurations record `loss_type` to distinguish new runs from older two-plane losses.

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

`--size` accepts width and height and defaults to `1920 1080`. The network pads dimensions to a multiple of 16 and crops the generated phase back to the requested size. `--distance-near`, `--distance-far`, `--depth-threshold`, and `--near-is` configure the two-plane mapping.

Generated images only go to `output/`. Weights, loss CSV files, and run configuration JSON files go to `checkpoint/`.
