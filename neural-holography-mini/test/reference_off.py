"""ASM sampling aligned to neural-holography-main/propagation_ASM.py.

Equations adapted from Peng et al., Neural Holography (2020), CC BY-NC 4.0.
Original source: ../../../../neural-holography-main/propagation_ASM.py.
No camera calibration is used. The existing equivalent U-Nets are reused.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F
from holonet import HoloNet


def propagation_asm(field, feature_size, wavelength, distance, kernel=None, return_kernel=False):
    height, width = field.shape[-2:]
    if height % 2 or width % 2:
        raise ValueError("Reference experiment requires even dimensions")
    ph, pw = height // 2, width // 2
    padded = F.pad(field, (pw, pw, ph, ph))
    if kernel is None:
        dy, dx = (feature_size, feature_size) if isinstance(feature_size, (int, float)) else feature_size
        ny, nx = padded.shape[-2:]
        y, x = dy * ny, dx * nx
        fy = np.linspace(-1 / (2 * dy) + 0.5 / (2 * y), 1 / (2 * dy) - 0.5 / (2 * y), ny)
        fx = np.linspace(-1 / (2 * dx) + 0.5 / (2 * x), 1 / (2 * dx) - 0.5 / (2 * x), nx)
        fx, fy = np.meshgrid(fx, fy)
        # Match original ordering: float64 NumPy root, float32 tensor, then z.
        exponent = torch.tensor(2 * math.pi * np.sqrt(1 / wavelength**2 - fx**2 - fy**2),
                                dtype=field.real.dtype, device=field.device) * distance
        limit_y = 1 / (wavelength * np.sqrt(1 + (2 * distance / y)**2))
        limit_x = 1 / (wavelength * np.sqrt(1 + (2 * distance / x)**2))
        mask = torch.tensor((np.abs(fx) < limit_x) & (np.abs(fy) < limit_y),
                            dtype=field.real.dtype, device=field.device)
        kernel = torch.fft.ifftshift(torch.complex(mask * exponent.cos(), mask * exponent.sin()), dim=(-2, -1))
    if return_kernel:
        return kernel
    spectrum = torch.fft.fft2(torch.fft.ifftshift(padded, dim=(-2, -1)), norm="ortho")
    output = torch.fft.fftshift(torch.fft.ifft2(spectrum * kernel, norm="ortho"), dim=(-2, -1))
    return output[..., ph:ph + height, pw:pw + width]


class ReferenceHoloNet(HoloNet):
    def forward(self, target_amplitude, depth=None, controlnet=None):
        if controlnet is not None:
            raise ValueError("Reference experiment supports off only")
        field = torch.polar(target_amplitude, self.initial_phase(target_amplitude))
        if not self.backward_kernel.numel():
            self.backward_kernel = propagation_asm(field, self.feature_size, self.wavelength,
                                                   -self.distance, return_kernel=True).detach()
        slm = propagation_asm(field, self.feature_size, self.wavelength, -self.distance,
                              kernel=self.backward_kernel)
        return self.final_phase_only(torch.cat((slm.abs(), slm.angle()), dim=1))
