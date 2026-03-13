import os
import sys
import unittest

import torch


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from chronologer_unimod_loss import LogL_Loss


class ChronologerUnimodLossTest(unittest.TestCase):
    def test_logl_loss_is_finite_and_deterministic(self):
        loss_fx = LogL_Loss(n_sources=2)

        pred = torch.tensor([[0.9], [1.1], [1.3]], dtype=torch.float32)
        true = torch.tensor([[1.0], [1.0], [1.5]], dtype=torch.float32)
        source = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=torch.float32,
        )

        loss_1 = loss_fx(pred, true, source)
        loss_2 = loss_fx(pred, true, source)

        self.assertTrue(torch.isfinite(loss_1).item())
        self.assertAlmostEqual(float(loss_1.item()), float(loss_2.item()), places=7)

    def test_logl_loss_backprop(self):
        loss_fx = LogL_Loss(n_sources=2)

        pred = torch.tensor([[0.8], [1.2]], dtype=torch.float32, requires_grad=True)
        true = torch.tensor([[1.0], [1.1]], dtype=torch.float32)
        source = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

        loss = loss_fx(pred, true, source)
        loss.backward()

        self.assertIsNotNone(pred.grad)
        self.assertIsNotNone(loss_fx.source_scale.weight.grad)


if __name__ == '__main__':
    unittest.main()

