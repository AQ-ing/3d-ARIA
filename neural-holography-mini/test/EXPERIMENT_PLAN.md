# 待手动运行：损失函数与初始化对照

目的：检查感知损失是否改善可见纹理，以及它训练出的 off 初始化是否有利于双深度。
本计划尚未运行，不预填实验结果。

| 组名 | 初始化 | 损失 | 步数 |
| --- | --- | --- | ---: |
| off_mse | 数据集 off epoch 040 | MSE | 800 |
| off_vgg-low | 同上 | vgg-low | 800 |
| joint_mse | off_mse | MSE | 1200 |
| controlnet-only_mse | off_mse | MSE | 1200 |
| joint_vgg-low | off_vgg-low | vgg-low | 1200 |
| controlnet-only_vgg-low | off_vgg-low | vgg-low | 1200 |

固定：0031 RGB/depth，960×640，绿色通道，全图，seed 42，Adam lr=1e-4。
保持现有双深度 0.18/0.22 m、阈值 0.1、bright 为 near；无增强、无 CITL。
这是损失与匹配初始化的组合对照，ControlNet 结果不能单独归因于损失变化。
所有组使用振幅 PSNR 和相同亮度匹配评价；感知损失数值不与 MSE 数值直接比较。
最终还需要检查树枝、文字细节与分区边缘，不能仅凭 PSNR 选优。

在 neural-holography-mini 下手动运行：

```powershell
conda run --no-capture-output -n e2e python -B test/run_experiments.py
```

自动串行运行六组，每次新建 test/experiments/<时间>/：

- 各组 .log：训练过程和错误信息。
- checkpoint/：最终权重（双深度包含 HoloNet 和 ControlNet，不含优化器）。
- output/：最终重建图片。
- manifest.json：已完成组的命令、初始化来源和退出码。
- results.json、report.md：整批结束后的汇总，全图及双深度分区 PSNR。

off 失败则跳过依赖它的两组，继续另一个损失分支；中途停止可通过已有日志定位进度。
脚本不支持断点接续，重新执行会创建新目录并重跑。运行完告知目录即可分析，无需逐组确认。
本计划取代 README 中旧的三学习率实验队列说明；历史结果仍保留。
