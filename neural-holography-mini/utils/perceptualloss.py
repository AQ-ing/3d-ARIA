import torch
import torchvision.models
import torch.nn.functional as F


class PerceptualLoss(torch.nn.modules.loss._Loss):
    """Frozen VGG19 feature loss plus pixel loss, averaged over the batch.

    Accepts single-channel amplitudes or RGB; matches the original project's
    feature inputs without extra normalization.
    """

    def __init__(self, pixel_loss=1.0, l1_loss=False, style_loss=0.0, lambda_feat=1, include_vgg_layers=('1', '2', '3', '4', '5')):
        super().__init__()

        # download pretrained vgg19 if necessary and instantiate it
        vgg19 = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1)
        self.vgg_layers = vgg19.features

        # the vgg feature layers we want to use for the perceptual loss
        self.layer_name_mapping = {
        }
        if '1' in include_vgg_layers:
            self.layer_name_mapping['3'] = "conv1_2"
        if '2' in include_vgg_layers:
            self.layer_name_mapping['8'] = "conv2_2"
        if '3' in include_vgg_layers:
            self.layer_name_mapping['13'] = "conv3_2"
        if '4' in include_vgg_layers:
            self.layer_name_mapping['22'] = "conv4_2"
        if '5' in include_vgg_layers:
            self.layer_name_mapping['31'] = "conv5_2"

        # VGG 仅提取固定特征，保留重建图到网络参数的梯度。
        last_layer = max(map(int, self.layer_name_mapping), default=-1)
        self.vgg_layers = self.vgg_layers[:last_layer + 1].requires_grad_(False).eval()

        # weights for pixel loss and style loss (feature loss assumed 1.0)
        self.pixel_loss = pixel_loss
        self.l1_loss = l1_loss
        self.lambda_feat = lambda_feat
        self.style_loss = style_loss

    def forward(self, input, target):
        if input.shape != target.shape or input.ndim != 4 or input.shape[1] not in (1, 3):
            raise ValueError('Expected matching [batch, 1 or 3, height, width] tensors')
        if input.shape[1] == 1:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        target = target.detach()

        # 像素项和特征项均按 batch、通道与空间取平均。
        lossValue = input.new_zeros(())
        if self.l1_loss:
            lossValue = lossValue + 2 * self.pixel_loss * F.l1_loss(input, target)
        elif self.pixel_loss:
            lossValue = lossValue + 2 * self.pixel_loss * F.mse_loss(input, target)

        x, y = input, target
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            # 目标分支不建立计算图；不改变一次处理的图像数量。
            with torch.no_grad():
                y = module(y)
            if name in self.layer_name_mapping:
                lossValue = lossValue + self.lambda_feat * F.mse_loss(x, y)

                # Gram matrix for style loss
                if self.style_loss:
                    A = x.flatten(2)
                    B = y.flatten(2)
                    G_A = A @ torch.transpose(A, 1, 2)
                    G_B = B @ torch.transpose(B, 1, 2)
                    scale = x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3]
                    lossValue = lossValue + self.style_loss * F.mse_loss(G_A, G_B, reduction='sum') / scale
        return lossValue
