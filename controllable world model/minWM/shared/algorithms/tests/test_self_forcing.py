"""Tests for self_forcing.py."""

import torch
import pytest
from algorithms.self_forcing import (
    sample_exit_step,
    create_gradient_mask,
    apply_gradient_mask,
)


class TestSampleExitStep:
    def test_same_across_blocks(self):
        """All blocks should get the same exit step."""
        result = sample_exit_step(5, 3, same_across_blocks=True)
        assert len(result) == 3
        assert result[0] == result[1] == result[2]

    def test_different_across_blocks(self):
        """Each block gets its own exit step."""
        result = sample_exit_step(5, 3, same_across_blocks=False)
        assert len(result) == 3
        for idx in result:
            assert 0 <= idx < 5

    def test_range(self):
        """Exit steps should be in [0, num_denoising_steps)."""
        for _ in range(50):
            result = sample_exit_step(4, 1)
            assert 0 <= result[0] < 4

    def test_single_step(self):
        """With 1 denoising step, exit must be 0."""
        result = sample_exit_step(1, 5)
        assert all(idx == 0 for idx in result)


class TestCreateGradientMask:
    def test_last_n_frames(self):
        """Mask should be 1 for last N frames, 0 elsewhere."""
        shape = [2, 10, 4, 8, 8]
        mask = create_gradient_mask(
            total_frames=10, last_n_frames=3,
            shape=shape, frame_dim=1,
            device=torch.device("cpu"),
        )
        assert mask.shape == torch.Size(shape)
        # First 7 frames should be 0
        assert (mask[:, :7] == 0).all()
        # Last 3 frames should be 1
        assert (mask[:, 7:] == 1).all()

    def test_all_frames(self):
        """When last_n_frames >= total_frames, all should be 1."""
        shape = [1, 5, 4, 8, 8]
        mask = create_gradient_mask(
            total_frames=5, last_n_frames=10,
            shape=shape, frame_dim=1,
            device=torch.device("cpu"),
        )
        assert (mask == 1).all()

    def test_zero_frames(self):
        """When last_n_frames=0, all should be 0."""
        shape = [1, 5, 4, 8, 8]
        mask = create_gradient_mask(
            total_frames=5, last_n_frames=0,
            shape=shape, frame_dim=1,
            device=torch.device("cpu"),
        )
        assert (mask == 0).all()

    def test_bool_dtype(self):
        shape = [1, 5, 4, 8, 8]
        mask = create_gradient_mask(
            total_frames=5, last_n_frames=2,
            shape=shape, frame_dim=1,
            device=torch.device("cpu"),
            dtype=torch.bool,
        )
        assert mask.dtype == torch.bool
        assert mask[:, 3:].all()
        assert not mask[:, :3].any()


class TestApplyGradientMask:
    def test_gradient_flows_through_mask(self):
        """Gradient should only flow where mask is 1."""
        video = torch.randn(1, 4, 2, 3, 3, requires_grad=True)
        mask = torch.zeros(1, 4, 2, 3, 3)
        mask[:, 2:] = 1.0  # grad only for last 2 frames
        result = apply_gradient_mask(video, mask)
        loss = result.sum()
        loss.backward()
        # Grad should be 0 for first 2 frames, 1 for last 2
        assert (video.grad[:, :2] == 0).all()
        assert (video.grad[:, 2:] == 1).all()

    def test_values_preserved(self):
        """Output values should match input regardless of mask."""
        video = torch.randn(1, 4, 2, 3, 3)
        mask = torch.zeros(1, 4, 2, 3, 3)
        mask[:, 2:] = 1.0
        result = apply_gradient_mask(video, mask)
        torch.testing.assert_close(result, video)

    def test_full_mask(self):
        """With mask=1 everywhere, gradient flows everywhere."""
        video = torch.randn(1, 4, 2, 3, 3, requires_grad=True)
        mask = torch.ones(1, 4, 2, 3, 3)
        result = apply_gradient_mask(video, mask)
        result.sum().backward()
        assert (video.grad == 1).all()
