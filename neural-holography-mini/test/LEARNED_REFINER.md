# 少步可学习相位修正模块

目标：保留冻结的单深度模型，在输出端附加共享的小网络。没有每张图独立训练的相位参数表。
所有改动仅在 test，本次只执行检查，没有启动正式训练。

## 结构与训练

基础模型输出 φ0，每次推理都从 φ0 开始，状态清零。重复 K=8 或16次：

1. 用两个深度 ASM 计算分区合成振幅及 MSE。
2. 计算损失对当前相位的梯度 g（不修改模型权重）。
3. 根据 g 更新一、二阶动量，得到 Adam 式方向 d。
4. 共享小网络输入归一化 g、d、sinφ、cosφ、步数比例、总步数编码（6通道）。
5. 1×1 Conv(6→8) → SiLU → 1×1 Conv(8→2)，输出局部增益 a 和修正 b。
6. φ_next = φ - 0.1 × [2 sigmoid(a) × d + 0.1 tanh(b)]。

共74个参数，各位置、各修正步共享。输出头零初始化，初始等价于固定 Adam 式更新。
不要求初始输出仍等于 φ0，因为这一模块设计就是执行固定数量的物理修正。
不对相位做 Hardtanh。模型接口只要求基础相位和可微目标函数，不访问 HoloNet 内部特征。
深度、目标图、波长等通过目标函数进入物理梯度；模块不依赖固定输入图片尺寸。

外层训练从头展开 K 步，用末步重建损失更新74个参数；不把上次训练相位带入下次。
为控制成本，将物理梯度及更新规则输入视为停止梯度的特征，不求 FFT Hessian，
但保留更新输出到共享参数的链式梯度。这是第一阶近似，不是精确的二阶展开训练。
基础模型冻结并缓存输出；每次推理仍会计算 K 次相位梯度。

这不是“把逐像素优化结果存成新网络”，也不是单次 CNN 前向；推理包含物理迭代。
目前只是单图验证。结构上可复用不等于权重已经跨图/跨网络泛化，后续必须另做多图验证。

## 本轮四组实验

| 组 | 单次推理修正步数 | 外层训练次数 | 作用 |
| --- | ---: | ---: | --- |
| fixed8 | 8 | 0 | 不学习的同计算路径对照 |
| fixed16 | 16 | 0 | 不学习的同计算路径对照 |
| learned8 | 8 | 200 | 学习更新规则 |
| learned16 | 16 | 200 | 学习更新规则 |

两种 learned 使用相同 seed，Adam 外层学习率1e-3；输入1920×1072、ROI1600×880、
0.18/0.22m、阈值0.1、绿色通道，仍以 current checkpoint 为基础。
200次外层训练分别包含约1600/3200次内部梯度计算，不能等同于200次普通相位优化。
是否值得使用学习模块，首先比较 learned 与对应 fixed 的质量和 evaluation_seconds；
若无收益，应报告无收益，不能只与旧CNN比较。单图训练之后的质量也不是验证集指标。

手动执行（不会自动降低分辨率）：

```powershell
conda run --no-capture-output -n e2e python -B test/run_phase_adapter.py --suite refiner
```

配置检查：

```powershell
conda run -n e2e python -B test/run_phase_adapter.py --suite refiner --dry-run
```

结果位于 test/experiments/<时间>_phase_adapter_refiner/。各组记录日志、config.json、
history.json、results.json、checkpoint/adapter.pth 和 output/mergerecon0031.png。
evaluation_seconds 是一次完整修正加末端评价耗时，不含基础模型推理；
elapsed_seconds 包含训练与周期评价。首次运行含预热因素，不可用于严格性能比较。
fixed 组也保存未使用的规则权重用于统一格式，实际 trainable_parameters=0；
checkpoint 不包含每图相位或动量状态。

## 加载使用

创建 LearnedPhaseRefiner，加载 phase_adapter 字段；给定新 base_phase 和目标函数 objective：

```python
model.eval()
with torch.no_grad():
    phase = model.refine(base_phase, objective, iterations=8, learned=True)
```

objective 必须计算标量振幅损失。不要用 torch.inference_mode 包裹，因为内部需要相位梯度。
推理不调用 optimizer，不更新网络参数。fixed 对照用 learned=False。
verify_phase_run.py 已支持重新加载这类 checkpoint 独立复算。

已检查：初始化与fixed结果一致、物理损失到规则参数的有限梯度、基础相位不获得梯度、
no_grad环境推理和保存加载一致。没有为验证而执行正式训练。
