import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_holonet import merged_amplitude_loss


class MergedAmplitudeLossTests(unittest.TestCase):
    def setUp(self):
        self.target = torch.tensor([[[[0.2, 0.4, 0.6, 0.8]]]], dtype=torch.float64)
        near = torch.tensor([[[[1., 1., 0., 0.]]]], dtype=torch.float64)
        self.masks = (near, 1 - near)

    def test_correct_selected_regions_fit_despite_unselected_light(self):
        near = torch.tensor([[[[0.4, 0.8, 30., 40.]]]], dtype=torch.float64)
        far = torch.tensor([[[[50., 60., 1.2, 1.6]]]], dtype=torch.float64)
        loss, scale, merged = merged_amplitude_loss((near, far), self.masks, self.target)
        torch.testing.assert_close(merged, 2 * self.target)
        self.assertAlmostEqual(scale.item(), 0.5)
        self.assertLess(loss.item(), 1e-28)

    def test_unselected_values_have_no_effect_or_direct_gradient(self):
        near = torch.tensor([[[[0.3, 0.9, 2., 3.]]]], dtype=torch.float64, requires_grad=True)
        far = torch.tensor([[[[4., 5., 0.8, 0.5]]]], dtype=torch.float64, requires_grad=True)
        loss, scale, merged = merged_amplitude_loss((near, far), self.masks, self.target)
        changed = (near.detach() + 100 * self.masks[1], far.detach() + 200 * self.masks[0])
        new_loss, new_scale, _ = merged_amplitude_loss(changed, self.masks, self.target)
        torch.testing.assert_close(loss, new_loss)
        torch.testing.assert_close(scale, new_scale)
        contributions = [((scale * merged - self.target).square() * m).mean() for m in self.masks]
        torch.testing.assert_close(sum(contributions), loss)
        loss.backward()
        for recon, mask in zip((near, far), self.masks):
            self.assertTrue(torch.isfinite(recon.grad).all())
            self.assertEqual(torch.count_nonzero(recon.grad * (1 - mask)).item(), 0)
            self.assertGreater(torch.count_nonzero(recon.grad * mask).item(), 0)

    def test_zero_reconstruction_is_finite(self):
        recon = torch.zeros_like(self.target, requires_grad=True)
        loss, scale, _ = merged_amplitude_loss((recon, recon), self.masks, self.target)
        torch.testing.assert_close(loss, self.target.square().mean())
        self.assertTrue(torch.isfinite(scale))
        loss.backward()
        self.assertTrue(torch.isfinite(recon.grad).all())


if __name__ == "__main__":
    unittest.main()
