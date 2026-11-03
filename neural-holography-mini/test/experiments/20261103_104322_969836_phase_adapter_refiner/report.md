# Phase adapter results

200 outer training steps (fixed baselines: 0); full-resolution ROI amplitude MSE.
Historical ControlNet-only MSE: 26.89 dB at 1200 steps. Pixel runs are per-image diagnostics, not reusable networks.

| Run | Status | PSNR | Parameters | Elapsed seconds | Evaluation seconds | Peak allocated MB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| fixed8 | completed | 26.811150389671017 | 0 | 0.8188971000054153 | 0.8188829999999143 | 690.333984375 |
| fixed16 | completed | 29.739594955091064 | 0 | 1.0832257000001846 | 1.08320310000272 | 690.369140625 |
| learned8 | completed | 26.654968802825604 | 74 | 81.43019460000505 | 0.34057520001078956 | 2218.3701171875 |
| learned16 | completed | 29.12866741090869 | 74 | 151.9316245999944 | 0.6405098000104772 | 3788.3076171875 |