"""Small CPU checks only. No optimizer or training run."""
import io
import torch
from phase_adapter import PhaseAdapter


torch.set_num_threads(2)
torch.manual_seed(42)
adapter = PhaseAdapter()
target = torch.rand(1, 1, 33, 49)
base = torch.randn_like(target)
mask = (target > 0.5).float()
offset = mask * 2 - 1
phase = adapter.phase(target, base, mask, offset)
torch.testing.assert_close(phase, base, atol=0, rtol=0)
phase.square().mean().backward()
assert adapter.output.weight.grad is not None
assert torch.isfinite(adapter.output.weight.grad).all()
assert adapter.output.weight.grad.abs().sum() > 0
# Verify periodic phase representation and upstream gradients once the zero head opens.
with torch.no_grad():
    adapter.output.weight.fill_(0.01)
adapter.zero_grad(set_to_none=True)
a = adapter.phase(target, base, mask, offset)
b = adapter.phase(target, base + 2 * torch.pi, mask, offset)
torch.testing.assert_close(torch.exp(1j * a), torch.exp(1j * b), atol=2e-6, rtol=2e-5)
a.square().mean().backward()
assert adapter.enc0[0].weight.grad.abs().sum() > 0
buffer = io.BytesIO()
torch.save(adapter.state_dict(), buffer)
buffer.seek(0)
restored = PhaseAdapter()
restored.load_state_dict(torch.load(buffer, weights_only=True))
torch.testing.assert_close(a, restored.phase(target, base, mask, offset))
print("PASS: zero initialization, gradients, arbitrary shape, phase periodicity, weight round-trip")
print("adapter_parameters=", sum(p.numel() for p in adapter.parameters()))

# Verify all new variants through the real two-plane propagation loss, without an optimizer.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from propagation import propagation_asm
from phase_adapter import build_adapter
base = torch.randn(1, 1, 64, 96)
target = torch.rand_like(base)
mask = (target > 0.5).float()
for kind in ("small", "wide", "context", "pixel"):
    variant = build_adapter(kind, base.shape)
    condition = variant.condition(target, base, mask, mask * 2 - 1)
    delta = variant(condition)
    torch.testing.assert_close(delta, torch.zeros_like(base), atol=0, rtol=0)
    field = torch.polar(torch.ones_like(base), base + delta)
    planes = [propagation_asm(field, 6.4e-6, 520e-9, z).abs() for z in (0.18, 0.22)]
    merged = planes[0] * mask + planes[1] * (1 - mask)
    scale = (merged * target).mean() / merged.square().mean().clamp_min(1e-8)
    (merged * scale - target).square().mean().backward()
    gradients = [p.grad for p in variant.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert sum(g.abs().sum() for g in gradients) > 0
    if kind == "context":
        assert variant.context.output.weight.grad.abs().sum() > 0
    print(kind, "parameters=", sum(p.numel() for p in variant.parameters()), "ASM gradient OK")
