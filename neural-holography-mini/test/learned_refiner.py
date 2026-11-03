"""Shared, image-independent update rule with first-order unrolled training."""
import torch
from torch import nn
from phase_adapter import PhaseAdapter


class LearnedPhaseRefiner(nn.Module):
    condition = staticmethod(PhaseAdapter.condition)

    def __init__(self):
        super().__init__()
        # Same small pointwise rule at all spatial positions and refinement steps.
        self.rule = nn.Sequential(nn.Conv2d(6, 8, 1), nn.SiLU(), nn.Conv2d(8, 2, 1))
        nn.init.zeros_(self.rule[-1].weight)
        nn.init.zeros_(self.rule[-1].bias)

    def refine(self, base_phase, objective, iterations=8, learned=True):
        phase = base_phase.detach()
        first = torch.zeros_like(phase)
        second = torch.zeros_like(phase)
        for index in range(1, iterations + 1):
            # Detach the physical gradient to avoid Hessians/second-order FFT graphs.
            # Preserve the update-rule graph through `phase` for first-order training.
            with torch.enable_grad():
                probe = phase.detach().requires_grad_(True)
                loss = objective(probe)
                gradient, = torch.autograd.grad(loss, probe)
            gradient = gradient.detach()
            first = 0.9 * first + 0.1 * gradient
            second = 0.999 * second + 0.001 * gradient.square()
            direction = (first / (1 - 0.9**index)) / ((second / (1 - 0.999**index)).sqrt() + 1e-8)
            if learned:
                rms = gradient.square().mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-12)
                features = torch.cat(((gradient / rms).clamp(-5, 5), direction,
                                      phase.detach().sin(), phase.detach().cos(),
                                      torch.full_like(phase, index / iterations),
                                      torch.full_like(phase, iterations / 16)), dim=1)
                gain, correction = self.rule(features).chunk(2, dim=1)
                update = 0.1 * ((2 * gain.sigmoid()) * direction + 0.1 * correction.tanh())
            else:
                update = 0.1 * direction
            phase = phase - update
        return phase
