import math

import torch
import torch.nn.functional as F


def propagation_asm(field, feature_size, wavelength, distance, kernel=None, return_kernel=False):
    height, width = field.shape[-2:]
    pad_h, pad_w = height // 2, width // 2
    padding = (pad_w, pad_w, pad_h, pad_h)
    padded = F.pad(field, padding)
    padded_height, padded_width = padded.shape[-2:]

    if kernel is None:
        dy, dx = (feature_size, feature_size) if isinstance(feature_size, (int, float)) else feature_size
        fy = torch.fft.fftfreq(padded_height, d=dy, device=field.device, dtype=field.real.dtype)
        fx = torch.fft.fftfreq(padded_width, d=dx, device=field.device, dtype=field.real.dtype)
        fy, fx = torch.meshgrid(fy, fx, indexing="ij")
        root = (1.0 / wavelength**2 - fx.square() - fy.square()).clamp_min(0).sqrt()
        extent_y, extent_x = dy * padded_height, dx * padded_width
        fy_limit = 1.0 / (wavelength * math.sqrt(1 + (2 * abs(distance) / extent_y) ** 2))
        fx_limit = 1.0 / (wavelength * math.sqrt(1 + (2 * abs(distance) / extent_x) ** 2))
        bandlimit = (fy.abs() < fy_limit) & (fx.abs() < fx_limit)
        kernel = torch.polar(bandlimit.to(field.real.dtype), 2 * math.pi * distance * root)

    if return_kernel:
        return kernel

    spectrum = torch.fft.fft2(torch.fft.ifftshift(padded, dim=(-2, -1)), norm="ortho")
    propagated = torch.fft.fftshift(torch.fft.ifft2(spectrum * kernel, norm="ortho"), dim=(-2, -1))
    return propagated[..., pad_h:pad_h + height, pad_w:pad_w + width]
