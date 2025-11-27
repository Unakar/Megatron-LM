"""Unit tests for MuonBall optimizer."""

import math
import torch
import pytest

try:
    from emerging_optimizers.orthogonalized_optimizers import MuonBall
    from emerging_optimizers.orthogonalized_optimizers.spectral_ball_utils import (
        compute_target_radius,
    )

    HAVE_MUON_BALL = True
except ImportError:
    HAVE_MUON_BALL = False


@pytest.mark.skipif(not HAVE_MUON_BALL, reason="MuonBall not available")
class TestMuonBall:
    """Test MuonBall optimizer functionality."""

    def test_basic_optimization(self):
        """Test basic MuonBall optimization step."""
        torch.manual_seed(42)

        # Create a simple 2D weight matrix
        W = torch.randn(128, 64, dtype=torch.float32)

        # Create optimizer
        opt = MuonBall(
            [W],
            lr=0.01,
            momentum_beta=0.9,
            use_nesterov=True,
            weight_decay=0.0,
            radius_mode='spectral_mup',
            retract_mode='hard',
            power_iteration_steps=10,
            msign_steps=5,
        )

        # Compute target radius for spectral_mup mode
        target_R = math.sqrt(128 / 64)  # sqrt(n_out / n_in)

        # Simulate one optimization step
        W.grad = torch.randn_like(W)

        # Get spectral norm before step
        sigma_before = torch.linalg.matrix_norm(W, ord=2).item()

        # Perform optimization step
        opt.step()

        # Get spectral norm after step
        sigma_after = torch.linalg.matrix_norm(W, ord=2).item()

        print(f"Before step: σ={sigma_before:.4f}")
        print(f"After step:  σ={sigma_after:.4f}")
        print(f"Target R:    {target_R:.4f}")
        print(f"Ratio σ/R:   {sigma_after/target_R:.4f}")

        # After hard retraction, spectral norm should be very close to R
        assert abs(sigma_after - target_R) / target_R < 0.01, (
            f"Spectral norm {sigma_after:.4f} not close to target {target_R:.4f}"
        )

    def test_retraction_maintains_radius(self):
        """Test that retraction maintains spectral radius over multiple steps."""
        torch.manual_seed(42)

        W = torch.randn(256, 128, dtype=torch.float32)

        opt = MuonBall(
            [W],
            lr=0.02,
            momentum_beta=0.95,
            use_nesterov=False,
            weight_decay=0.0,
            radius_mode='spectral_mup',
            retract_mode='hard',
            power_iteration_steps=10,
            msign_steps=5,
        )

        target_R = math.sqrt(256 / 128)  # sqrt(2) ≈ 1.414

        # Run multiple optimization steps
        num_steps = 10
        for i in range(num_steps):
            W.grad = torch.randn_like(W) * 0.1
            opt.step()

            sigma = torch.linalg.matrix_norm(W, ord=2).item()
            ratio = sigma / target_R

            print(f"Step {i+1}: σ={sigma:.4f}, R={target_R:.4f}, ratio={ratio:.4f}")

            # Check that spectral norm stays close to R
            assert abs(sigma - target_R) / target_R < 0.01, (
                f"Step {i+1}: Spectral norm {sigma:.4f} drifted from target {target_R:.4f}"
            )

    def test_identity_radius_mode(self):
        """Test MuonBall with identity radius mode (R=1)."""
        torch.manual_seed(42)

        W = torch.randn(100, 50, dtype=torch.float32)

        opt = MuonBall(
            [W],
            lr=0.01,
            momentum_beta=0.9,
            radius_mode='identity',  # R = 1.0
            retract_mode='hard',
        )

        # Perform step
        W.grad = torch.randn_like(W)
        opt.step()

        sigma = torch.linalg.matrix_norm(W, ord=2).item()
        print(f"σ={sigma:.4f}, expected R=1.0")

        # Should be close to 1.0
        assert abs(sigma - 1.0) < 0.01, f"Spectral norm {sigma:.4f} not close to 1.0"

    def test_different_shapes(self):
        """Test MuonBall on different matrix shapes."""
        torch.manual_seed(42)

        shapes = [
            (512, 256),  # sqrt(512/256) = sqrt(2) ≈ 1.414
            (256, 512),  # sqrt(256/512) = sqrt(0.5) ≈ 0.707
            (1024, 1024),  # sqrt(1) = 1.0
            (2048, 512),  # sqrt(4) = 2.0
        ]

        for shape in shapes:
            W = torch.randn(*shape, dtype=torch.float32)

            opt = MuonBall(
                [W],
                lr=0.01,
                radius_mode='spectral_mup',
                retract_mode='hard',
            )

            # Expected target radius
            target_R = math.sqrt(shape[0] / shape[1])

            W.grad = torch.randn_like(W)
            opt.step()

            sigma = torch.linalg.matrix_norm(W, ord=2).item()
            ratio = sigma / target_R

            print(f"Shape {shape}: σ={sigma:.4f}, R={target_R:.4f}, ratio={ratio:.4f}")

            assert abs(sigma - target_R) / target_R < 0.01, (
                f"Shape {shape}: σ={sigma:.4f} not close to R={target_R:.4f}"
            )

    @pytest.mark.parametrize("use_nesterov", [True, False])
    def test_nesterov_momentum(self, use_nesterov):
        """Test MuonBall with and without Nesterov momentum."""
        torch.manual_seed(42)

        W = torch.randn(128, 64, dtype=torch.float32)

        opt = MuonBall(
            [W],
            lr=0.01,
            momentum_beta=0.9,
            use_nesterov=use_nesterov,
            radius_mode='spectral_mup',
            retract_mode='hard',
        )

        # Run a few steps
        for _ in range(5):
            W.grad = torch.randn_like(W) * 0.1
            opt.step()

        # Just check that it runs without error and maintains radius
        target_R = math.sqrt(128 / 64)
        sigma = torch.linalg.matrix_norm(W, ord=2).item()

        print(f"Nesterov={use_nesterov}: σ={sigma:.4f}, R={target_R:.4f}")

        assert abs(sigma - target_R) / target_R < 0.01


if __name__ == "__main__":
    # Run tests directly
    if HAVE_MUON_BALL:
        test = TestMuonBall()
        print("Testing basic optimization...")
        test.test_basic_optimization()
        print("\nTesting radius maintenance...")
        test.test_retraction_maintains_radius()
        print("\nTesting identity radius mode...")
        test.test_identity_radius_mode()
        print("\nTesting different shapes...")
        test.test_different_shapes()
        print("\nTesting Nesterov momentum...")
        test.test_nesterov_momentum(True)
        test.test_nesterov_momentum(False)
        print("\n✓ All tests passed!")
    else:
        print("MuonBall not available, skipping tests")
