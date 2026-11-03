# 从 current checkpoint 训练 ControlNet-only

初始化：test/experiments/20261102_102239_off_reference/checkpoint/current.pth
（单深度 0.2 m，ROI 振幅 PSNR 31.128668 dB）。不使用 reference checkpoint。

加载 HoloNet 后完整复制其最终相位 U-Net 到 ControlNet；条件卷积和残差输出卷积零初始化。
冻结 HoloNet 的参数和 BN 统计量，仅优化 ControlNet。现有 train_single.py 已支持这些逻辑。

保持与 current 相同的绿色通道、1920×1072 输入、homography 几何处理、
1600×880 中央 ROI、vgg-low 与最小二乘亮度匹配。RGB/depth 使用相同缩放裁剪和补零。
双深度为 0.18/0.22 m，depth > 0.1 为近景；先按分区合成振幅再计算 ROI 损失。
新增训练 1200 步，Adam lr=1e-4，seed=42。学习率低于从零训练 current 的 1e-3。
使用现有 ASM，不启用 --off-reference。

在 neural-holography-mini 下手动执行：

```powershell
conda run --no-capture-output -n e2e python -B test/run_control_from_current.py
```

每次创建 test/experiments/<时间>_control_from_current/：
manifest.json 记录完整命令，train.log 记录过程及错误，results.json 汇总退出码、
全 ROI 振幅 MSE/PSNR 和近远区域 PSNR。output/ 仅放最终 mergerecon 图片；
checkpoint/ 保存 HoloNet 和 ControlNet 两部分权重，不含优化器状态。
不自动降低尺寸或更改训练设置，不覆盖旧实验。执行失败也会保存退出码和日志。

31.13 dB 是单深度基线，双深度合成指标用于观察质量差距，不代表两个任务物理约束相同，
也不能单凭合成图证明各平面的离焦/串扰效果。
本次只准备实验，没有运行训练。
