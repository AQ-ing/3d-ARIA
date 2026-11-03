# Off 与原项目对齐

仅在 test 中实现，没有修改主目录，也没有启动训练。

已检查原项目 train_holonet.py、holonet.py、propagation_ASM.py 和 U-Net 工具实现。
当前简化 U-Net 的四层下采样、16 初始通道、最内层无 BN、跳接、转置卷积和
Hardtanh 均与原项目训练配置对应；不能仅凭文件更短判断网络容量不足。

主要对齐项：

- ASM：按原项目 linspace 频率网格、NumPy 双精度根号计算、转 float32 后乘距离，
  以及相同带限和中心裁剪。正反向都使用同一实现，缓存传播核。
- 数据：1920×1072，先按原 loader 的 homography 方式缩放/裁剪到 1600×880，
  再补零到输入尺寸；损失只看中央区域。
- 原项目训练目标：重建振幅乘 sqrt(0.95) 后计算 vgg-low，取消训练时逐图最小二乘缩放。
- 从零训练 lr=1e-3；原项目载入权重后会除以 10。本轮为两组相同 seed 的随机初始化。
- 保持用户要求，不做数据增强；当前仍是单图实验，不是原项目完整数据集复现。

## 手动实验

```powershell
conda run --no-capture-output -n e2e python -B test/run_off_reference.py
```

两组各 1000 步，均 off，且共享上述分辨率、ROI、vgg-low、seed、随机初始化和学习率：

1. current：当前 ASM + 训练时最小二乘亮度匹配。
2. reference：原项目 ASM + 固定强度系数 0.95。

这是传播与亮度损失的组合对照；如果改善明显，再拆分两项做单因素实验。
不会自动减小分辨率，失败信息（包括显存不足）写入日志；两组均记录退出码。
本轮计算量高于此前 960×640；运行完成后提供目录即可读取分析。

输出在 test/experiments/<时间>_off_reference：output 中只有图片，checkpoint 中保存权重，
日志和 JSON 在运行目录。权重仍兼容当前 HoloNet 参数结构，但 reference 权重推理必须使用
reference_off.ReferenceHoloNet 和对应 ASM，不能直接用主目录 eval 得到同样重建。

最终 amplitude_psnr 为亮度匹配后的 ROI 振幅 PSNR，供两组同口径比较；
reference 另记录 reference_fixed_scale_psnr，反映实际训练亮度口径。
PNG 采用匹配亮度并量化，指标在量化前计算。ROI 分数不能与之前全图分数直接比较。
未证实达到用户原项目的高 PSNR，需本轮实际结果；也需要明确原项目高 PSNR 的评价口径。

## 已执行的非训练检查

64×96 complex64 随机输入，与原项目 propagation_ASM 的输出最大误差为 0；
传播梯度有限、ReferenceHoloNet 前向尺寸正确。没有执行优化器更新。
