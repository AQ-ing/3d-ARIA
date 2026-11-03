# 下一批 ControlNet 消融（手动执行）

所有组从同一 current.pth（off ROI 31.13 dB）重新初始化。
复用已完成的 controlnet-only/vgg-low、lr=1e-4、0.18/0.22 m、1200 步实验
作为 23.80 dB 基线，不重复跑它。固定 1920×1072、1600×880 ROI、绿色通道、
homography、阈值 0.1、seed 42；不自动降低尺寸，不改主目录。

| 组 | 与基线相比改变 | 步数 | 要回答的问题 |
| --- | --- | ---: | --- |
| same_plane | 两个平面都为 0.2 m | 300 | 控制分支加入后能否保留已学会的单深度重建？ |
| joint_vgg | 解冻 HoloNet | 1200 | 冻结主网络是否限制双深度适配？ |
| only_mse | 损失改为 MSE | 1200 | 相同初始化下感知损失是否影响像素还原？ |
| joint_mse | 解冻 HoloNet，并用 MSE | 1200 | 联合训练与损失的组合效果；与 only_mse 配对比较 |
| only_small_gap | 深度改为 0.195/0.205 m | 1200 | 较小深度间距是否更易适配？ |
| only_lr3e4 | 学习率改为 3e-4 | 1200 | 原学习率是否导致固定预算内适配过慢？ |

共 6300 个优化步，串行运行；全分辨率较耗时。same_plane 不作为真正的双深度结果，
两个分区在这个诊断中只是同一平面的不同像素。小间距组也改变了任务难度，不能直接宣称模型更优。
joint 组显存可能更高，若失败记录错误并继续下一组，不悄悄改变实验条件。

```powershell
conda run --no-capture-output -n e2e python -B test/run_control_ablation.py
```

训练步骤同步输出到终端与各组日志。输出目录：test/experiments/<时间>_control_ablation/。
results.json 每组更新，包含完整命令、退出码及全 ROI/近远区域 PSNR；全部结束后生成 report.md。
checkpoint/ 存两分支权重，output/ 存各组图片。重复运行建立新目录；不支持中途断点恢复。
批量脚本现在会尝试在运行期间关闭传统 CMD 的快速编辑模式，退出时恢复原设置。
已在运行的旧进程不会自动应用代码更新，可在窗口属性中手动关闭快速编辑，避免重跑。
如仍进入文字选择模式，按 Esc 取消选择；该处理不控制 Windows Terminal 自身的选择行为。

只检查路径和打印计划、不训练：

```powershell
conda run -n e2e python -B test/run_control_ablation.py --dry-run
```

结果解读须结合细节与分区边缘，PSNR 基于匹配亮度的振幅而非 RGB。
当前仍只输出合成图，不用于证明两个实际平面的离焦或串扰质量。
