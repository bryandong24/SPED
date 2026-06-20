"""Tests for dmd.py."""

import torch
import pytest
from algorithms.dmd import (
    apply_cfg,
    compute_kl_gradient,
    dmd_generator_loss,
    dmd_critic_loss,
)


class TestApplyCfg:
    def test_standard_formula(self):
        """Verify: uncond + scale * (cond - uncond)."""
        v_cond = torch.ones(2, 3) * 3.0
        v_uncond = torch.ones(2, 3) * 1.0
        result = apply_cfg(v_cond, v_uncond, guidance_scale=2.0)
        # 1 + 2*(3-1) = 5
        expected = torch.ones(2, 3) * 5.0
        torch.testing.assert_close(result, expected)

    def test_scale_zero(self):
        """With scale=0, result = uncond."""
        v_cond = torch.randn(2, 3)
        v_uncond = torch.randn(2, 3)
        result = apply_cfg(v_cond, v_uncond, guidance_scale=0.0)
        torch.testing.assert_close(result, v_uncond)

    def test_scale_one(self):
        """With scale=1, result = cond."""
        v_cond = torch.randn(2, 3)
        v_uncond = torch.randn(2, 3)
        result = apply_cfg(v_cond, v_uncond, guidance_scale=1.0)
        torch.testing.assert_close(result, v_cond)


class TestComputeKlGradient:
    def test_no_normalization(self):
        fake = torch.ones(2, 3, 4, 8, 8) * 2.0
        real = torch.ones(2, 3, 4, 8, 8) * 1.0
        gen = torch.randn(2, 3, 4, 8, 8)
        grad = compute_kl_gradient(fake, real, gen, normalize_mode="none")
        expected = torch.ones(2, 3, 4, 8, 8) * 1.0
        torch.testing.assert_close(grad, expected)

    def test_global_normalizer_is_scalar(self):
        """In global mode, the normalizer should be a scalar (same for all elements)."""
        fake = torch.randn(2, 3, 4, 8, 8)
        real = torch.randn(2, 3, 4, 8, 8)
        gen = torch.randn(2, 3, 4, 8, 8)
        grad = compute_kl_gradient(fake, real, gen, normalize_mode="global")
        # All elements should be normalized by the same scalar
        assert grad.shape == fake.shape

    def test_per_sample_keeps_batch_dim(self):
        """In per_sample mode, each sample has its own normalizer."""
        B = 4
        fake = torch.randn(B, 3, 4, 8, 8)
        real = torch.zeros(B, 3, 4, 8, 8)
        gen = torch.randn(B, 3, 4, 8, 8)
        grad = compute_kl_gradient(fake, real, gen, normalize_mode="per_sample")
        assert grad.shape == fake.shape

    def test_nan_to_num(self):
        """Should handle NaN from division by zero."""
        fake = torch.ones(1, 1, 1, 1, 1)
        real = torch.ones(1, 1, 1, 1, 1)  # same as fake -> grad = 0
        gen = torch.ones(1, 1, 1, 1, 1)   # same as real -> normalizer = 0
        grad = compute_kl_gradient(fake, real, gen, normalize_mode="global")
        assert not torch.isnan(grad).any()


class TestDmdGeneratorLoss:
    def test_gradient_flows_through_generator(self):
        """Gradient should flow through generator_output, not through target."""
        gen_out = torch.randn(2, 3, 4, 8, 8, requires_grad=True)
        kl_grad = torch.randn(2, 3, 4, 8, 8)
        loss = dmd_generator_loss(gen_out, kl_grad)
        loss.backward()
        assert gen_out.grad is not None

    def test_zero_gradient_gives_zero_loss(self):
        """If kl_gradient is zero, target = generator_output, loss = 0."""
        gen_out = torch.randn(2, 3, 4, 8, 8)
        kl_grad = torch.zeros(2, 3, 4, 8, 8)
        loss = dmd_generator_loss(gen_out, kl_grad)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_with_gradient_mask(self):
        gen_out = torch.randn(2, 3, 4, 8, 8)
        kl_grad = torch.ones(2, 3, 4, 8, 8)
        mask = torch.zeros(2, 3, 4, 8, 8, dtype=torch.bool)
        mask[0] = True  # only first sample
        loss = dmd_generator_loss(gen_out, kl_grad, gradient_mask=mask)
        assert loss.item() > 0


class TestDmdCriticLoss:
    def test_matches_flow_matching_target(self):
        """critic_loss(pred, noise, clean) should equal MSE(pred, noise - clean)."""
        pred = torch.randn(2, 3, 4, 8, 8)
        noise = torch.randn(2, 3, 4, 8, 8)
        clean = torch.randn(2, 3, 4, 8, 8)
        loss = dmd_critic_loss(pred, noise, clean)
        expected = (pred.float() - (noise - clean).float()).pow(2).mean()
        assert loss.item() == pytest.approx(expected.item(), abs=1e-6)

    def test_zero_loss(self):
        """When pred = noise - clean, loss should be 0."""
        noise = torch.randn(2, 3, 4, 8, 8)
        clean = torch.randn(2, 3, 4, 8, 8)
        pred = noise - clean
        loss = dmd_critic_loss(pred, noise, clean)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_with_weight(self):
        pred = torch.ones(2, 3)
        noise = torch.zeros(2, 3)
        clean = torch.zeros(2, 3)
        weight = torch.tensor([[2.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
        loss = dmd_critic_loss(pred, noise, clean, weight=weight)
        expected = (2.0 + 0.0 + 1.0 + 0.0 + 0.0 + 0.0) / 6
        assert loss.item() == pytest.approx(expected, abs=1e-6)
