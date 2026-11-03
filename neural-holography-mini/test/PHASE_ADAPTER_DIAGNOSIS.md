# 第二轮：相位修正可达质量与网络表达诊断

目标仍是冻结单深度模型、通过输出端附加模块实现多深度。仅修改 test，未启动训练。
本轮统一 1200 步，都是同一份 current checkpoint 和基础相位，1920×1072、
1600×880 ROI、MSE、seed=42、0.18/0.22 m。无增强，无额外光学模型改动。

| 组 | 模型/变量 | 学习率 | 目的 |
| --- | --- | --- | --- |
| pixel_lr003 | 每像素独立 Δφ | 0.03 | 测试不受 CNN 表达限制时是否容易优化 |
| pixel_lr01 | 每像素独立 Δφ | 0.1 | 避免直接相位诊断受单一学习率误导 |
| small_1200 | 原 8/16/32 通道适配器 | 1e-3 | 检查更长预算收益，提供本轮基线 |
| wide_1200 | 16/32/64 通道适配器 | 1e-3 | 检查增大容量是否有帮助 |
| context_1200 | 原适配器＋1/8 尺寸上下文分支 | 1e-3 | 检查扩大空间覆盖是否有帮助 |

上下文分支将同样的五通道条件平均池化到 1/8 后处理，再上采样残差与高分辨率残差相加。
两个输出头均零初始化，起始相位完全等于基础相位。平均池化后的正余弦是粗尺度特征，
不是重新构建的低分辨率物理相位。该分支扩大感受范围，并非真正全局传播算子。

每像素相位优化是单图诊断，不能泛化到新图，不能作为“低成本可复用附加网络”交付，
也不保证找到物理最优解。若它表现好而 CNN 差，才有较强理由继续优化 CNN 表达/训练；
若它也差，应进一步排查损失、优化设置和物理约束，不能直接认定任务不可能。

## 一次运行

```powershell
conda run --no-capture-output -n e2e python -B test/run_phase_adapter.py --suite diagnosis
```

总计 6000 步，串行执行；每组记录每 100 步 PSNR/耗时、参数量、显存及最终重建。
独立目录 test/experiments/<时间>_phase_adapter_diagnosis/，日志同步打印。
失败会记录并继续后续组，不自动缩小分辨率；加宽网络可能占用更多显存。
small_1200 从相同初始化重跑 1200 步，不是加载上轮 600 步权重；避免丢失 Adam 状态影响对照。
不与历史实验宣称相同墙钟时间，结果需按 history.json 的质量-时间曲线比较。

检查配置而不训练：

```powershell
conda run -n e2e python -B test/run_phase_adapter.py --suite diagnosis --dry-run
```

新 checkpoint 的 config.adapter_kind 标记模型类型；推理应调用
build_adapter(kind, base_phase.shape) 并加载 phase_adapter 字段。
small 的类和默认参数不变，旧 checkpoint 仍可通过 PhaseAdapter() 加载。
pixel 权重 delta 只适用于训练图像与尺寸；其输出可直接与对应基础相位相加。
报告中的 PSNR 仍为分区合成后的 ROI 振幅指标，不验证非目标平面串扰。
