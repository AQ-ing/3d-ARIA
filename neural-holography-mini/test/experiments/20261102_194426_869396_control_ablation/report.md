# ControlNet ablation

Baseline: off 31.128668 dB; prior control-only vgg 23.801595 dB.
same_plane is a diagnostic, not a two-depth result.

| Run | Steps | PSNR | Near | Far | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| same_plane | 300 | 31.427231 | 32.2 | 30.02 | completed |
| joint_vgg | 1200 | 26.095553 | 27.54 | 23.88 | completed |
| only_mse | 1200 | 26.888556 | 27.98 | 25.06 | completed |
| joint_mse | 1200 | 29.387612 | 30.54 | 27.5 | completed |
| only_small_gap | 1200 | 25.745128 | 27.7 | 23.08 | completed |
| only_lr3e4 | 1200 | 25.128366 | 26.52 | 22.97 | completed |