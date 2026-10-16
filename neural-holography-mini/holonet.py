import math

import torch
from torch import nn

from propagation import propagation_asm


class Conv2dSame(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, bias=bias)

    def forward(self, x):
        return self.conv(self.pad(x))


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm=nn.BatchNorm2d):
        super().__init__()
        layers = [nn.ReflectionPad2d(1), nn.Conv2d(in_channels, in_channels, 3, bias=norm is None)]
        if norm is not None:
            layers.append(norm(in_channels, affine=True))
        layers += [nn.LeakyReLU(0.2, inplace=True), nn.ReflectionPad2d(1),
                   nn.Conv2d(in_channels, out_channels, 4, stride=2, bias=norm is None)]
        if norm is not None:
            layers.append(norm(out_channels, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm=nn.BatchNorm2d):
        super().__init__()
        layers = [nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1,
                                     bias=norm is None)]
        if norm is not None:
            layers.append(norm(out_channels, affine=True))
        layers += [nn.ReLU(inplace=True), Conv2dSame(out_channels, out_channels, bias=norm is None)]
        if norm is not None:
            layers.append(norm(out_channels, affine=True))
        layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x, skip=None):
        if skip is not None:
            x = torch.cat((skip, x), dim=1)
        return self.net(x)


class UnetSkip(nn.Module):
    def __init__(self, outer_channels, inner_channels, submodule=None, norm=nn.BatchNorm2d):
        super().__init__()
        self.down = DownBlock(outer_channels, inner_channels, norm=norm)
        self.submodule = submodule
        self.up = UpBlock(inner_channels * (2 if submodule else 1), outer_channels, norm=norm)

    def forward(self, x):
        y = self.down(x)
        if self.submodule is not None:
            y = self.submodule(y)
        y = self.up(y)
        return torch.cat((x, y), dim=1)


class Unet(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=16, depth=4, max_channels=256):
        super().__init__()
        self.input = nn.Sequential(
            Conv2dSame(in_channels, base_channels),
            nn.BatchNorm2d(base_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        inner = min(2 ** (depth - 1) * base_channels, max_channels)
        block = UnetSkip(inner, inner, norm=None)
        for level in reversed(range(depth - 1)):
            outer = min(2 ** level * base_channels, max_channels)
            inner = min(2 ** (level + 1) * base_channels, max_channels)
            block = UnetSkip(outer, inner, block)
        self.body = block
        self.output = Conv2dSame(base_channels * 2, out_channels, bias=True)

    def forward(self, x):
        return self.output(self.body(self.input(x)))


class InitialPhaseUnet(nn.Module):
    def __init__(self, base_channels=16, depth=4):
        super().__init__()
        self.net = nn.Sequential(Unet(1, 1, base_channels, depth), nn.Hardtanh(-math.pi, math.pi))

    def forward(self, amplitude):
        return self.net(amplitude)


class FinalPhaseOnlyUnet(nn.Module):
    def __init__(self, base_channels=16, depth=4):
        super().__init__()
        self.net = nn.Sequential(Unet(2, 1, base_channels, depth), nn.Hardtanh(-math.pi, math.pi))

    def forward(self, amplitude_phase):
        return self.net(amplitude_phase)


class HoloNet(nn.Module):
    """HoloNet phase generator with a differentiable ASM backward propagation."""

    def __init__(self, wavelength=520e-9, feature_size=6.4e-6, distance=0.2,
                 base_channels=16, depth=4):
        super().__init__()
        self.wavelength = wavelength
        self.feature_size = feature_size
        self.distance = distance
        self.initial_phase = InitialPhaseUnet(base_channels, depth)
        self.final_phase_only = FinalPhaseOnlyUnet(base_channels, depth)
        self.register_buffer("backward_kernel", torch.empty(0), persistent=False)

    def forward(self, target_amplitude):
        initial_phase = self.initial_phase(target_amplitude)
        target_field = torch.polar(target_amplitude, initial_phase)

        if self.backward_kernel.numel() == 0:
            self.backward_kernel = propagation_asm(
                target_field, self.feature_size, self.wavelength, -self.distance, return_kernel=True
            ).detach()

        slm_field = propagation_asm(
            target_field, self.feature_size, self.wavelength, -self.distance,
            kernel=self.backward_kernel,
        )
        amplitude_phase = torch.cat((slm_field.abs(), slm_field.angle()), dim=1)
        return self.final_phase_only(amplitude_phase)
