"""Small forward/backward checks; no training or optimizer steps."""
import sys
from pathlib import Path
import torch
from learned_refiner import LearnedPhaseRefiner
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from propagation import propagation_asm

torch.set_num_threads(2)
torch.manual_seed(42)
base = torch.randn(1, 1, 32, 48)
target = torch.rand_like(base)
mask = (target > 0.5).float()


def objective(phase):
    field = torch.polar(torch.ones_like(phase), phase)
    planes = [propagation_asm(field, 6.4e-6, 520e-9, z).abs() for z in (0.18, 0.22)]
    merged = planes[0] * mask + planes[1] * (1-mask)
    scale = (merged * target).mean() / merged.square().mean().clamp_min(1e-8)
    return (merged * scale - target).square().mean()


model = LearnedPhaseRefiner()
with torch.no_grad():
    fixed = model.refine(base, objective, 3, learned=False)
    learned = model.refine(base, objective, 3)
torch.testing.assert_close(fixed, learned, atol=0, rtol=0)
assert not learned.requires_grad
phase = model.refine(base, objective, 3)
objective(phase).backward()
assert base.grad is None
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
assert model.rule[-1].weight.grad.abs().sum() > 0
# Inference must neither update weights nor need an optimizer.
saved = {k: v.clone() for k, v in model.state_dict().items()}
restored = LearnedPhaseRefiner()
restored.load_state_dict(saved)
with torch.no_grad():
    result = restored.refine(base, objective, 3)
torch.testing.assert_close(result, learned)
for k, v in model.state_dict().items():
    torch.testing.assert_close(v, saved[k], atol=0, rtol=0)
print("PASS: zero-head equals fixed update, ASM gradient reaches shared rule, base frozen, no-grad inference, reload")
print("shared_parameters=", sum(p.numel() for p in model.parameters()))
