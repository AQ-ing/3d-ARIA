# HoloNet / Depth ControlNet

默认数据集为 `D:/wy/3d-ARIA/data/DIV2K_train_HR`，深度为同名的 `DIV2K_train_HR_depth`。默认 **1920×1080、batch size 8、20 epochs**，一次直接处理 8 张，不使用梯度累积。

训练直接调用 `utils.augmented_image_loader.ImageLoader`：关闭水平/垂直翻转，通过 `direct_resize=True` 直接缩放，不裁剪、不额外补零。双深度模式在同一个 loader 中按文件名读取配对深度图，随同一份顺序打乱，避免 RGB/depth 错配。RGB 转线性强度后开平方得到振幅，深度只除以 255。

## 训练

```powershell
conda run --no-capture-output -n e2e python train_holonet.py --runid baseline01

conda run --no-capture-output -n e2e python train_holonet.py --runid joint01 --controlnet-mode joint

conda run --no-capture-output -n e2e python train_holonet.py --runid control01 --controlnet-mode controlnet-only --init-checkpoint checkpoint/baseline01_off_epoch_020.pth
```

`off` 为单深度 HoloNet（0.20 m）；`joint` 共同训练；`controlnet-only` 冻结 HoloNet 权重及 BN 状态，只训练复制的最终相位 U-Net 分支。

训练脚本只训练，不输出相位或重建图片。每个 epoch 保存一次权重：

- `checkpoint/<runid>_<mode>_epoch_001.pth`：同一文件包含 HoloNet、ControlNet（启用时）及推理配置。
- `checkpoint/<runid>_config.json`：运行配置。
- `checkpoint/tensorboard/<runid>/`：TensorBoard 日志，记录 `Loss/train_step`、`Loss/train_epoch` 和 `Learning_rate`；epoch 损失按实际图像数量平均。

`runid` 默认生成时间名称，也可手动设置；已有同名运行会报错，防止覆盖。checkpoint 不包含 Adam 状态。`--init-checkpoint` 仅用于初始化 HoloNet，同时支持旧的纯 HoloNet 权重和新的合并 checkpoint。

```powershell
tensorboard --logdir checkpoint/tensorboard
```

双深度默认近平面 0.18 m、远平面 0.22 m，阈值 0.1，亮区对应近平面。损失调用 `utils.perceptualloss.PerceptualLoss(lambda_feat=0.025)`，与原项目默认 vgg-low 一致：`2 × 像素 MSE + 0.025 × 各层 VGG19 特征 MSE 之和`。双深度先按掩膜拼接重建振幅；单深度直接使用重建振幅，每张图独立计算亮度系数，再与完整目标比较。单通道振幅复制为三通道输入 VGG，VGG 权重冻结，损失按 batch 取平均。预训练 VGG19 权重由 torchvision 缓存加载，首次使用若未缓存则自动下载。网络内部补齐至 16 的倍数，输出裁剪回设定尺寸。

## 推理

```powershell
# 单深度 HoloNet
conda run -n e2e python eval_holonet.py --controlnet-mode off --checkpoint checkpoint/baseline01_off_epoch_020.pth --image data/image/chinese.png --output output/baseline01

# 双深度 ControlNet，单独训练 ControlNet 的权重使用 controlnet-only 模式
conda run -n e2e python eval_holonet.py --controlnet-mode joint --checkpoint checkpoint/joint01_joint_epoch_020.pth --image data/image/chinese.png --depth data/depth/chinese.png --output output/joint01
```

`eval_holonet.py` 只执行推理，不加载 VGG、不计算损失或评估指标、不读取训练 JSON。显式指定 `--controlnet-mode off/joint/controlnet-only`，分辨率、通道和传播参数从 checkpoint 内读取；启用 ControlNet 时必须传入深度。`--image` / `--depth` 可以是单张文件，也可以是目录。

每张输入只输出一张图片：单深度保存为 `<指定output>/recon<图像名>.png`，双深度保存为 `<指定output>/mergerecon<图像名>.png`，图像名不含原扩展名。显示重建图时按重建振幅的最大值归一化，两个平面共用比例。

旧纯权重仍可推理：默认采用 1920×1080、绿色通道和当前默认传播参数；双深度需保留配对的 `*_holonet.pth` / `*_controlnet.pth` 文件。非默认设置的旧权重需要转换为带相应 config 的新 checkpoint。

数据读取复用 `utils.srgb_gamma2lin`。
