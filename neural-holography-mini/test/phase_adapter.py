"""Model-independent phase residual adapter. All inputs live on the SLM grid.

Target/mask maps are conditioning signals, not propagated complex fields.
The base phase may be supplied by any compatible single-plane generator.
"""
import torch
from torch import nn
import torch.nn.functional as F


def block(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1),
                         nn.GroupNorm(4, out_channels), nn.SiLU(),
                         nn.Conv2d(out_channels, out_channels, 3, padding=1),
                         nn.GroupNorm(4, out_channels), nn.SiLU())


class PhaseAdapter(nn.Module):
    """Small two-scale U-Net with full-resolution skip and zero residual head."""
    def __init__(self, width=8):
        super().__init__()
        self.enc0 = block(5, width)
        self.enc1 = block(width, 2 * width)
        self.middle = block(2 * width, 4 * width)
        self.dec1 = block(6 * width, 2 * width)
        self.dec0 = block(3 * width, width)
        self.output = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def condition(target, base_phase, near_mask, distance_offset):
        """distance_offset is (assigned distance - base distance) / 0.02 m."""
        return torch.cat((target, base_phase.sin(), base_phase.cos(), near_mask, distance_offset), dim=1)

    def forward(self, condition):
        high = self.enc0(condition)
        low = self.enc1(F.avg_pool2d(high, 2))
        mid = self.middle(F.avg_pool2d(low, 2))
        low = self.dec1(torch.cat((low, F.interpolate(mid, size=low.shape[-2:], mode="bilinear", align_corners=False)), 1))
        high = self.dec0(torch.cat((high, F.interpolate(low, size=high.shape[-2:], mode="bilinear", align_corners=False)), 1))
        return self.output(high)

    def phase(self, target, base_phase, near_mask, distance_offset):
        # No hard clipping: exp(i*phase) already implements phase periodicity.
        return base_phase + self(self.condition(target, base_phase, near_mask, distance_offset))


class ContextPhaseAdapter(PhaseAdapter):
    """Full-resolution path plus a second adapter at 1/8 resolution."""
    def __init__(self):
        super().__init__()
        self.context = PhaseAdapter()

    def forward(self, condition):
        local = super().forward(condition)
        coarse = F.avg_pool2d(condition, 8, ceil_mode=True)
        coarse = self.context(coarse)
        return local + F.interpolate(coarse, size=condition.shape[-2:], mode="bilinear", align_corners=False)


class PixelPhaseResidual(nn.Module):
    """Per-image optimization diagnostic, not a reusable image-to-phase network."""
    condition = staticmethod(PhaseAdapter.condition)

    def __init__(self, shape):
        super().__init__()
        self.delta = nn.Parameter(torch.zeros(shape))

    def forward(self, condition):
        return self.delta


def build_adapter(kind, shape):
    if kind == "small":
        return PhaseAdapter()
    if kind == "wide":
        return PhaseAdapter(width=16)
    if kind == "context":
        return ContextPhaseAdapter()
    if kind == "pixel":
        return PixelPhaseResidual(shape)
    raise ValueError(f"Unknown adapter: {kind}")
