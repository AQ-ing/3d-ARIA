# 独立深度分区预览（草稿）

本脚本只读取 RGB 和深度图、保存分区图片，不加载模型，不训练，也不修改主训练代码或已有训练输出。需要 NumPy 和 OpenCV（现有 e2e 环境已安装）。

在 `neural-holography-mini` 目录执行：

```powershell
conda run -n e2e python -B drafts/depth_preview/partition_depth.py --threshold 0.05
```

所有相对输入、输出路径均相对于 `neural-holography-mini`，与命令当前目录无关。

```powershell
conda run -n e2e python -B drafts/depth_preview/partition_depth.py --image data/image/drink.png --depth data/depth/drink.png --threshold 0.05 --compare-thresholds 0.02 0.05 0.10 0.50 --size 1920 1080 --output output/depth_threshold_comparison_drink
```

深度图按灰度读取，除以 255 后双线性缩放，最后按固定阈值切分：分区 1 为 `D <= threshold`，分区 2 为 `D > threshold`。不使用百分位、最小最大归一化或形态学处理。`0.05` 约等于原始灰度 `12.75/255`，不是 5% 的面积。OpenCV 浮点双线性缩放与训练中的 PyTorch 缩放在阈值边界可能存在微小数值差异。

默认输出到单独的 `output/depth_threshold_comparison_chinese/`：

- `threshold_comparison.png`：各阈值对比。
- `selected_partition_preview.png`：`--threshold` 指定阈值的详细预览。
- `t*/mask_plane1.png`、`mask_plane2.png`：互补的二值掩膜，白色为选中区域。
- `t*/rgb_plane1.png`、`rgb_plane2.png`：对应的彩色区域，供观察分区，不是训练中的单通道振幅目标。

这里的 plane 1/2 仅表示暗区/其余区，尚未决定对应哪一个物理传播距离；深度黑白方向应结合深度估计方法确认。主训练代码的阈值仍保持原值，等待确认后再合并。
