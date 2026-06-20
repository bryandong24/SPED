"""Tests for ode_regression.py."""

import torch
import pytest
from algorithms.ode_regression import prepare_ode_input, ode_regression_loss


class TestPrepareOdeInput:
    def test_output_shapes(self):
        B, N, F, C, H, W = 2, 5, 3, 16, 8, 8
        traj = torch.randn(B, N, F, C, H, W)
        steps = list(range(N))
        noisy, clean, target, ts = prepare_ode_input(traj, steps, B, traj.device)
        assert noisy.shape == (B, F, C, H, W)
        assert clean.shape == (B, F, C, H, W)
        assert target.shape == (B, F, C, H, W)
        assert ts.shape == (B, F)

    def test_clean_is_last(self):
        B, N, F, C, H, W = 1, 4, 2, 4, 4, 4
        traj = torch.randn(B, N, F, C, H, W)
        _, clean, _, _ = prepare_ode_input(traj, list(range(N)), B, traj.device)
        torch.testing.assert_close(clean, traj[:, -1])

    def test_target_is_second_to_last(self):
        B, N, F, C, H, W = 1, 4, 2, 4, 4, 4
        traj = torch.randn(B, N, F, C, H, W)
        _, _, target, _ = prepare_ode_input(traj, list(range(N)), B, traj.device)
        torch.testing.assert_close(target, traj[:, -2])

    def test_timestep_in_range(self):
        B, N, F, C, H, W = 3, 6, 2, 4, 4, 4
        traj = torch.randn(B, N, F, C, H, W)
        steps = [100, 200, 300, 400, 500, 600]
        _, _, _, ts = prepare_ode_input(traj, steps, B, traj.device)
        # Timestep should be from steps[0:5] (excluding clean)
        for val in ts.flatten().tolist():
            assert val in steps[:N - 1]


class TestOdeRegressionLoss:
    def test_zero_loss(self):
        x = torch.randn(2, 3, 4, 8, 8)
        loss = ode_regression_loss(x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_positive_loss(self):
        pred = torch.ones(2, 3, 4, 8, 8)
        target = torch.zeros(2, 3, 4, 8, 8)
        loss = ode_regression_loss(pred, target)
        assert loss.item() == pytest.approx(1.0, abs=1e-6)

    def test_with_mask(self):
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.zeros(3)
        mask = torch.tensor([True, False, True])
        loss = ode_regression_loss(pred, target, mask=mask)
        expected = (1.0 + 9.0) / 2
        assert loss.item() == pytest.approx(expected, abs=1e-6)

    def test_with_weight(self):
        pred = torch.tensor([1.0, 1.0])
        target = torch.zeros(2)
        weight = torch.tensor([2.0, 0.0])
        loss = ode_regression_loss(pred, target, weight=weight)
        expected = (2.0 + 0.0) / 2
        assert loss.item() == pytest.approx(expected, abs=1e-6)
