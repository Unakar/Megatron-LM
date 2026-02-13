"""Unit tests for MuonHyperball optimizer."""

import math
import torch
import pytest
import sys
sys.path.append("/n/home07/nabreu/megatron/Megatron-LM")

try:
    from emerging_optimizers.orthogonalized_optimizers.muon_hyperball import MuonHyperball

    HAVE_MUON_HYPERBALL = True
except ImportError:
    HAVE_MUON_HYPERBALL = False


@pytest.mark.skipif(not HAVE_MUON_HYPERBALL, reason="MuonHyperball not available")
class TestMuonHyperball:
    """Test MuonHyperball optimizer functionality."""

    def test_basic_optimization(self):
        """Test basic MuonHyperball optimization step."""
        torch.manual_seed(42)

        W = torch.randn(128, 64, dtype=torch.float32)
        R_init = torch.norm(W, p="fro").item()

        opt = MuonHyperball(
            [W],
            lr=0.01,
            momentum_beta=0.9,
            use_nesterov=True,
            weight_decay=0.0,
            radius_mode='initialize',
            msign_steps=5,
        )

        W.grad = torch.randn_like(W)
        W_before = W.clone()

        opt.step()

        # Weights should have changed
        assert not torch.allclose(W, W_before), "Weights did not change after step"

        # Frobenius norm should be preserved
        R_after = torch.norm(W, p="fro").item()
        print(f"R_init={R_init:.4f}, R_after={R_after:.4f}, ratio={R_after/R_init:.6f}")
        assert abs(R_after - R_init) / R_init < 0.01, (
            f"Frobenius norm {R_after:.4f} not close to initial {R_init:.4f}"
        )

    def test_frobenius_norm_maintained(self):
        """Test that Frobenius norm is maintained over multiple steps."""
        torch.manual_seed(42)

        W = torch.randn(256, 128, dtype=torch.float32)
        R_init = torch.norm(W, p="fro").item()

        opt = MuonHyperball(
            [W],
            lr=0.02,
            momentum_beta=0.95,
            use_nesterov=False,
            weight_decay=0.0,
            radius_mode='initialize',
            msign_steps=5,
        )

        num_steps = 10
        for i in range(num_steps):
            W.grad = torch.randn_like(W) * 0.1
            opt.step()

            R = torch.norm(W, p="fro").item()
            ratio = R / R_init

            print(f"Step {i+1}: R={R:.4f}, R_init={R_init:.4f}, ratio={ratio:.6f}")

            assert abs(R - R_init) / R_init < 0.01, (
                f"Step {i+1}: Frobenius norm {R:.4f} drifted from initial {R_init:.4f}"
            )

    def test_frobenius_mup_radius_mode(self):
        """Test MuonHyperball with frobenius_mup radius mode (R=sqrt(n_out))."""
        torch.manual_seed(42)

        W = torch.randn(100, 50, dtype=torch.float32)
        target_R = math.sqrt(100)  # sqrt(n_out) = 10.0

        opt = MuonHyperball(
            [W],
            lr=0.01,
            momentum_beta=0.9,
            radius_mode='frobenius_mup',
            msign_steps=5,
        )

        W.grad = torch.randn_like(W)
        opt.step()

        R = torch.norm(W, p="fro").item()
        print(f"R={R:.4f}, target_R={target_R:.4f}")

        assert abs(R - target_R) / target_R < 0.01, (
            f"Frobenius norm {R:.4f} not close to target {target_R:.4f}"
        )

    def test_identity_radius_mode(self):
        """Test MuonHyperball with identity radius mode (R=1)."""
        torch.manual_seed(42)

        W = torch.randn(100, 50, dtype=torch.float32)

        opt = MuonHyperball(
            [W],
            lr=0.01,
            momentum_beta=0.9,
            radius_mode='identity',
            msign_steps=5,
        )

        W.grad = torch.randn_like(W)
        opt.step()

        R = torch.norm(W, p="fro").item()
        print(f"R={R:.4f}, expected R=1.0")

        # Tolerance is looser here because R=1.0 is small relative to msign update magnitude;
        # the retraction at the start of the NEXT step will fully correct the drift.
        assert abs(R - 1.0) < 0.02, f"Frobenius norm {R:.4f} not close to 1.0"

    def test_different_shapes(self):
        """Test MuonHyperball on different matrix shapes."""
        torch.manual_seed(42)

        shapes = [
            (512, 256),
            (256, 512),
            (1024, 1024),
            (2048, 512),
        ]

        for shape in shapes:
            W = torch.randn(*shape, dtype=torch.float32)
            R_init = torch.norm(W, p="fro").item()

            opt = MuonHyperball(
                [W],
                lr=0.01,
                radius_mode='initialize',
                msign_steps=5,
            )

            W.grad = torch.randn_like(W)
            opt.step()

            R_after = torch.norm(W, p="fro").item()
            ratio = R_after / R_init

            print(f"Shape {shape}: R_init={R_init:.4f}, R_after={R_after:.4f}, ratio={ratio:.6f}")

            assert abs(R_after - R_init) / R_init < 0.01, (
                f"Shape {shape}: Frobenius norm {R_after:.4f} not close to {R_init:.4f}"
            )

    @pytest.mark.parametrize("use_nesterov", [True, False])
    def test_nesterov_momentum(self, use_nesterov):
        """Test MuonHyperball with and without Nesterov momentum."""
        torch.manual_seed(42)

        W = torch.randn(128, 64, dtype=torch.float32)
        R_init = torch.norm(W, p="fro").item()

        opt = MuonHyperball(
            [W],
            lr=0.01,
            momentum_beta=0.9,
            use_nesterov=use_nesterov,
            radius_mode='initialize',
            msign_steps=5,
        )

        for _ in range(5):
            W.grad = torch.randn_like(W) * 0.1
            opt.step()

        R_after = torch.norm(W, p="fro").item()
        print(f"Nesterov={use_nesterov}: R_init={R_init:.4f}, R_after={R_after:.4f}")

        assert abs(R_after - R_init) / R_init < 0.01


if __name__ == "__main__":
    if HAVE_MUON_HYPERBALL:
        test = TestMuonHyperball()
        print("Testing basic optimization...")
        test.test_basic_optimization()
        print("\nTesting Frobenius norm maintenance...")
        test.test_frobenius_norm_maintained()
        print("\nTesting frobenius_mup radius mode...")
        test.test_frobenius_mup_radius_mode()
        print("\nTesting identity radius mode...")
        test.test_identity_radius_mode()
        print("\nTesting different shapes...")
        test.test_different_shapes()
        print("\nTesting Nesterov momentum...")
        test.test_nesterov_momentum(True)
        test.test_nesterov_momentum(False)
        print("\n✓ All tests passed!")
    else:
        print("MuonHyperball not available, skipping tests")
