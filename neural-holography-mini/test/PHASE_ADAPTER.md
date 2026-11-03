# 输出端轻量相位适配器

只添加 test 文件，不改主目录网络，不启动训练。原 ControlNet 实验仍可使用。

接口：base_phase 由任意兼容单深度模型产生，形状为 B×1×H×W、单位为弧度。
PhaseAdapter 接收目标振幅、sin(base_phase)、cos(base_phase)、近景 mask、归一化距离偏移，
输出 Δφ，最终相位 base_phase + Δφ。网络为两次下采样的 8/16/32 通道小 U-Net，
完整分辨率跳接，GroupNorm，不使用 BatchNorm 或 Hardtanh，输出卷积零初始化。
共 30,225 参数；现有复制 U-Net 的 ControlNet 共 1,537,285 参数，约相差 50.9 倍。
这只是参数量比较，不代表运行速度也提高 50 倍。

本次仍是固定 0.18/0.22 m 双深度。条件距离为 (像素分配距离 - 0.2)/0.02。
深度 mask 和目标图是空间条件，不能将它们视为 SLM 面传播后的物理量。
跨网络只需提供相位，但必须匹配网格、单位、波长、像素间距及基础深度。
同一份权重跨模型/图像/深度免训练复用尚未验证。

训练器复用现有 HoloNet 和 checkpoint/current.pth（31.13 dB 单深度基线）。
固定模型 eval 并关闭参数梯度，只计算一次基础相位，然后释放模型；缓存条件和 ASM 核。
基础相位缓存只对固定输入有效，数据集训练需要逐图生成/缓存，不能跨图复用。
保持 1920×1072 输入、1600×880 ROI、绿色通道、homography、阈值 0.1、seed 42。
训练使用分区合成振幅 MSE 与全 ROI 最小二乘亮度匹配。

## 手动运行

```powershell
conda run --no-capture-output -n e2e python -B test/run_phase_adapter.py
```

三组均为 600 步：

| 组 | 基础相位 | 学习率 | 目的 |
| --- | --- | --- | --- |
| adapter_lr1e3 | 冻结模型输出 | 1e-3 | 较快适配 |
| adapter_lr1e4 | 相同输出 | 1e-4 | 更保守优化 |
| zero_base_lr1e3 | 全零 | 1e-3 | 同参数同 seed 下，验证基础相位是否有益 |

零基础组同时将输入相位和输出相位基底替换为零，不能仅称“去掉相位输入”。
为保证代码路径与随机初始化一致，该组也执行一次基础推理后丢弃结果；因此不以其准备耗时
作为无需基础模型方案的效率基准。
这三组预算小于历史 ControlNet 的 1200 步；历史指标用于参考，不是等预算速度测试。

每次新建 test/experiments/<时间>_phase_adapter/，同步打印和记录日志，失败后继续下一组。
各组目录记录 config.json、history.json（0/100/…/600 步）、results.json；
checkpoint/adapter.pth 只保存适配器及配置，output/ 保存最终 mergerecon0031.png。
总目录生成 results.json 和 report.md。旧文件不覆盖，不支持中途断点恢复。
时间记录包含周期性评价及日志开销，setup_seconds 单独记录基础推理/缓存准备；
peak_allocated_mb 是缓存建立后训练阶段 PyTorch 分配显存峰值，不是整机显存或初始化峰值。

## 推理接口

先按保存配置获得 target、base_phase、near_mask、distance_offset（相同尺寸和设备）。
加载 adapter.pth 的 phase_adapter 字段到 PhaseAdapter，然后调用：

```python
adapter.eval()
with torch.no_grad():
    final_phase = adapter.phase(target, base_phase, near_mask, distance_offset)
```

需要原始单深度模型及权重来提供 base_phase；适配器 checkpoint 不复制基础权重。
零基础消融推理时也必须使用零 base_phase。关闭适配器时直接使用基础相位。
当前主目录 eval 不支持此新权重格式。

已执行小尺寸非训练检查：零残差严格等于基础相位、输出层与上游梯度有限、
奇数尺寸输出一致、相位加 2π 保持光场一致、权重保存加载一致；以及批量脚本 dry-run。
