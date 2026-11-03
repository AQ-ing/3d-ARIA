# Single-image overfitting

2026-11-03 的零训练深度迁移验证见 [DEPTH_PHYSICS_VALIDATION.md](DEPTH_PHYSICS_VALIDATION.md)。
包含传播公式、载波相位对齐、透镜相位、双平面串扰和完整结果路径；主目录代码未改动。

Run from `neural-holography-mini` using the current HoloNet, ControlNet, ASM,
ImageLoader and perceptual loss. The script repeatedly trains on one RGB image.
Depth is used only for ControlNet modes. It does not save optimizer states or
TensorBoard logs. Optional `--save-model` saves final weights (both branches in
ControlNet modes); this is initialization/inference data, not a resumable checkpoint.
After the last step it writes one amplitude PNG under
`test/output/<mode>/`.

```powershell
conda run -n e2e python test/train_single.py --image data/image/0031.png --steps 1000

conda run -n e2e python test/train_single.py --image data/image/0031.png --depth data/depth/0031.png --controlnet-mode joint --steps 1000

conda run -n e2e python test/train_single.py --image data/image/0031.png --depth data/depth/0031.png --controlnet-mode controlnet-only --init-checkpoint checkpoint/20261028_142836_541774_off_epoch_040.pth --steps 1000
```

`off` writes `test/output/off/recon-0031.png`; the other modes write
`test/output/<mode>/mergerecon-0031.png`. Use `--output test/output/<run-name>`
to keep multiple runs of the same mode separate. `--output` must stay inside
`test/`.
Defaults are 1920x1080, green channel, learning rate 1e-4, and 1000 steps.
The `controlnet-only` mode copies the loaded HoloNet's final U-Net into its
ControlNet branch, freezes HoloNet, and trains only ControlNet.

## 单图实验记录

只修改 test 内文件，复用主目录网络、loader、ASM 和损失工具。
输入为 0031，绿色通道；统一 seed=42，960×640，direct，全图评价。
双深度为 0.18/0.22 m，depth > 0.1 为近景（66.6%），其余为远景。
指标是未经 PNG 截断/量化的振幅 PSNR（峰值 1），先做全图最小二乘亮度匹配。
不是 RGB/sRGB PSNR；PNG 也用同一亮度匹配，未做 sRGB 编码。

| 实验 | 单图优化步数 | 全图 PSNR | 近景区域 PSNR | 远景区域 PSNR |
| --- | ---: | ---: | ---: | ---: |
| off，MSE，初次微调 | 1000 | 25.05 | 26.22 | 23.38 |
| off，MSE，追加低学习率微调 | 1000+1000 | 25.52 | 26.80 | 23.74 |
| joint，从上述 off 初始化 | 额外 1200 | 24.87 | 26.36 | 22.89 |
| controlnet-only，从上述 off 初始化 | 额外 1200 | 24.28 | 25.59 | 22.46 |

off 的分区指标也在 0.2 m 单平面重建上统计，方便比较同一批像素；并不代表 off 具有双深度能力。
所有实验均已有数据集预训练初始化，步数不包括预训练；两种 ControlNet 都完整复制最终 U-Net 并使用零卷积。
这些是单图、单 seed 实验。joint 距 off 仍差 0.65 dB，controlnet-only 差 1.24 dB。
局部预览显示树枝细节仍偏软，不能宣称已经达到原项目质量或完全解决双深度问题。
合成图指标也不验证非目标深度区域的串扰。

此前探索：1024×576、500 步时，vgg-low 为 23.34 dB，MSE 为 24.19 dB；
960×640、500 步 MSE 为 24.46 dB。不同尺寸结果不可作为严格消融。
homography/ROI 的 24.69 dB 改变了目标及评价区域，不与全图结果直接比较。

原项目训练脚本采用 1920×1072 输入、中央 1600×880 损失区域，默认 vgg-low，
固定强度系数 0.95；这里的全图最小二乘亮度匹配与它不同。
所检查的原项目 train_holonet.py 将 source_amplitude、zernike_coeffs、target_field、
latent_codes 设为 None，前向监督用 ASM，因此不能直接把当前差距归因于缺少 CITL。
分辨率、有效孔径、ROI、网络与传播数值实现均未完全对齐；尚未运行严格同条件的原项目基线。

## 复现

一键批量实验：`conda run --no-capture-output -n e2e python -B test/run_experiments.py`。
串行比较三种 off 学习率，再选最优 off 权重运行 joint 和 controlnet-only。
每次创建独立的 test/experiments/<时间>/，保存命令、日志、权重、图片；
全队列完成后统一写 results.json 和 report.md。失败项保留退出码及日志。
本轮 20261102_093044 的三个 off 在日志两位小数精度下均为 25.58 dB，
因此使用队列首个候选（1e-5）初始化双深度；不能据此判断该学习率更优。
后续训练日志提高 PSNR 精度以减少舍入造成的并列。

从 neural-holography-mini 运行，顺序执行：

```powershell
conda run -n e2e python -B test/train_single.py --image data/image/0031.png --init-checkpoint checkpoint/20261028_142836_541774_off_epoch_040.pth --size 960 640 --loss-region full --loss mse --steps 1000 --output test/output/experiments/off_mse_960x640_1000 --save-model test/checkpoint/off_mse_960x640_1000.pth

conda run -n e2e python -B test/train_single.py --image data/image/0031.png --init-checkpoint test/checkpoint/off_mse_960x640_1000.pth --size 960 640 --loss-region full --loss mse --learning-rate 0.00005 --steps 1000 --output test/output/experiments/off_refine_1000 --save-model test/checkpoint/off_refine_1000.pth

conda run -n e2e python -B test/train_single.py --image data/image/0031.png --depth data/depth/0031.png --controlnet-mode joint --init-checkpoint test/checkpoint/off_refine_1000.pth --size 960 640 --loss-region full --loss mse --steps 1200 --output test/output/experiments/joint_refined_off_1200 --save-model test/checkpoint/joint_refined_off_1200.pth

conda run -n e2e python -B test/train_single.py --image data/image/0031.png --depth data/depth/0031.png --controlnet-mode controlnet-only --init-checkpoint test/checkpoint/off_refine_1000.pth --size 960 640 --loss-region full --loss mse --steps 1200 --output test/output/experiments/control_refined_off_1200 --save-model test/checkpoint/control_refined_off_1200.pth

conda run -n e2e python -B test/compare_results.py
```

最后一条只加载权重推理，输出 test/results.json 与 test/output/comparison.png。
`--init-checkpoint` 只加载 HoloNet，ControlNet 总是重新复制初始化；不代表恢复整个实验。
`--preprocess homography --loss-region roi` 用于另行探索原项目的裁剪方式。
