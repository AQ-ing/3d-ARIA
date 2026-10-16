# HoloNet mini

This minimal project overfits HoloNet to one image. It does not require a pretrained checkpoint and does not include SGD, GS, DPAC, or CITL training code.

Run from this directory with the existing `e2e` environment:

```powershell
conda run -n e2e python train_holonet.py --image data/1.png --channel 1 --steps 20
```

The script uses CUDA when available and falls back to CPU. `--channel` selects red (`0`), green (`1`), or blue (`2`). `--size` takes width and height and defaults to `1920 1080`; for example, use `--size 1024 1024` for a square output. The network pads internally to a multiple of 16 and crops the phase back to the requested dimensions. Results are written to `output/`: `phase.png`, `reconstruction.png`, `holonet.pth`, and `loss.csv`.

This is a small single-image overfit smoke test, not a reproduction of the original training pipeline or a quality benchmark.

The sample image `data/1.png` is copied from the Neural Holography project and is distributed under CC BY.
