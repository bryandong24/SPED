"""Tests for flow_matching.py."""

import torch
import pytest
from algorithms.flow_matching import (
    add_flow_noise,
    flow_matching_target,
    pred_x0_from_flow,
    flow_matching_loss,
)


class TestAddFlowNoise:
    def test_sigma_zero_returns_clean(self):
        clean = torch.randn(2, 3, 4)
        noise = torch.randn_like(clean)
        sigma = torch.zeros(2, 1, 1)
        result = add_flow_noise(clean, noise, sigma)
        torch.testing.assert_close(result, clean)

    def test_sigma_one_returns_noise(self):
        clean = torch.randn(2, 3, 4)
        noise = torch.randn_like(clean)
        sigma = torch.ones(2, 1, 1)
        result = add_flow_noise(clean, noise, sigma)
        torch.testing.assert_close(result, noise)

    def test_interpolation(self):
        clean = torch.ones(1, 1, 1)
        noise = torch.zeros(1, 1, 1)
        sigma = torch.tensor([[[0.3]]])
        result = add_flow_noise(clean, noise, sigma)
        expected = 0.7 * clean + 0.3 * noise
        torch.testing.assert_close(result, expected)

    def test_video_shape(self):
        # [B, F, C, H, W]
        clean = torch.randn(2, 5, 16, 30, 52)
        noise = torch.randn_like(clean)
        sigma = torch.rand(2, 5, 1, 1, 1)
        result = add_flow_noise(clean, noise, sigma)
        assert result.shape == clean.shape


class TestFlowMatchingTarget:
    def test_basic(self):
        clean = torch.tensor([1.0, 2.0, 3.0])
        noise = torch.tensor([4.0, 5.0, 6.0])
        target = flow_matching_target(clean, noise)
        expected = torch.tensor([3.0, 3.0, 3.0])
        torch.testing.assert_close(target, expected)


class TestPredX0FromFlow:
    def test_roundtrip(self):
        """add_flow_noise -> flow_matching_target -> pred_x0_from_flow should recover clean."""
        clean = torch.randn(2, 3, 4, dtype=torch.float32)
        noise = torch.randn_like(clean)
        sigma = torch.rand(2, 1, 1).clamp(min=0.01)
        noisy = add_flow_noise(clean, noise, sigma)
        target = flow_matching_target(clean, noise)
        recovered = pred_x0_from_flow(noisy, target, sigma)
        torch.testing.assert_close(recovered, clean, atol=1e-5, rtol=1e-5)

    def test_bf16_precision(self):
        """pred_x0_from_flow should upcast to compute_dtype for stability."""
        clean = torch.randn(2, 3, 4, dtype=torch.bfloat16)
        noise = torch.randn_like(clean)
        sigma = torch.rand(2, 1, 1, dtype=torch.bfloat16).clamp(min=0.01)
        noisy = add_flow_noise(clean, noise, sigma)
        target = flow_matching_target(clean, noise)
        recovered = pred_x0_from_flow(noisy, target, sigma, compute_dtype=torch.float32)
        assert recovered.dtype == torch.bfloat16
        torch.testing.assert_close(recovered.float(), clean.float(), atol=0.05, rtol=0.05)


class TestFlowMatchingLoss:
    def test_zero_loss(self):
        x = torch.randn(2, 3)
        loss = flow_matching_loss(x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_known_value(self):
        pred = torch.tensor([1.0, 2.0])
        target = torch.tensor([0.0, 0.0])
        loss = flow_matching_loss(pred, target)
        expected = (1.0 + 4.0) / 2
        assert loss.item() == pytest.approx(expected, abs=1e-6)

    def test_with_weight(self):
        pred = torch.tensor([1.0, 2.0])
        target = torch.zeros(2)
        weight = torch.tensor([2.0, 0.5])
        loss = flow_matching_loss(pred, target, weight=weight)
        expected = (2.0 * 1.0 + 0.5 * 4.0) / 2
        assert loss.item() == pytest.approx(expected, abs=1e-6)

    def test_with_mask(self):
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.zeros(3)
        mask = torch.tensor([True, False, True])
        loss = flow_matching_loss(pred, target, mask=mask)
        expected = (1.0 + 9.0) / 2
        assert loss.item() == pytest.approx(expected, abs=1e-6)

    def test_reduction_none(self):
        pred = torch.tensor([1.0, 2.0])
        target = torch.zeros(2)
        loss = flow_matching_loss(pred, target, reduction="none")
        assert loss.shape == (2,)
