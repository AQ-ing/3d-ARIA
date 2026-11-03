# Batch experiment results

960x640, green amplitude, full-image evaluation, seed 42.
MSE versus vgg-low; both off runs use the same dataset-pretrained checkpoint.
Control modes start from their matching-loss off weights; steps exclude pretraining.

| Run | Additional steps | PSNR (dB) | Near | Far | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| off_mse | 800 | 24.873306 | None | None | 0 |
| joint_mse | 1200 | 24.739856 | 26.16 | 22.82 | 0 |
| controlnet-only_mse | 1200 | 23.941628 | 25.16 | 22.21 | 0 |
| off_vgg-low | 800 | 23.769248 | None | None | 0 |
| joint_vgg-low | 1200 | 22.756442 | 23.91 | 21.09 | 0 |
| controlnet-only_vgg-low | 1200 | 22.285139 | 23.31 | 20.77 | 0 |
