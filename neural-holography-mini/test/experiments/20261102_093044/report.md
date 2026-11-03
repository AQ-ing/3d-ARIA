# Batch experiment results

960x640, green amplitude, full-image MSE, seed 42.
All off candidates start from the same previously refined weights (2000 single-image steps).
Control modes start from the best candidate; listed steps are additional training.

| Run | Additional steps | PSNR (dB) | Near | Far | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| off_lr_1e-05 | 400 | 25.58 | None | None | 0 |
| off_lr_5e-05 | 400 | 25.58 | None | None | 0 |
| off_lr_0.0001 | 400 | 25.58 | None | None | 0 |
| joint | 1200 | 24.88 | 26.36 | 22.9 | 0 |
| controlnet-only | 1200 | 24.26 | 25.56 | 22.44 | 0 |
