# 完成后的分析

六组均正常完成。已查看 only_mse 和 joint_mse 输出，后者细节恢复更充分。

| 组 | PSNR dB |
| --- | ---: |
| 同深度诊断，300 步 | 31.43 |
| joint + vgg-low | 26.10 |
| controlnet-only + MSE | 26.89 |
| joint + MSE | 29.39 |
| controlnet-only + 小间距 | 25.75 |
| controlnet-only + lr=3e-4 | 25.13 |

除诊断外均 1200 步。历史基线 controlnet-only + vgg-low 为 23.80 dB，off 为 31.13 dB。
同深度可以保留并略改善 off，不支持“控制分支加入即破坏基础输出”的解释。
固定初始化后，MSE 提高 controlnet-only 约 3.09 dB；在 MSE 下解冻主网络再提高约 2.50 dB。
联合训练和 MSE 都有帮助，但 joint 与用户冻结基础模型、低成本复用的目标不完全一致。
小深度间距和更大学习率均改善有限；小间距改变任务，不能当作原双深度性能提升。

下一步采用输出端轻量相位适配器，先测 MSE；基础相位对照用于验证已有模型是否提供收益。
当前结果为单图单 seed，主要指标为 ROI 振幅 PSNR；不证明跨图/跨模型泛化或实际两平面串扰质量。
