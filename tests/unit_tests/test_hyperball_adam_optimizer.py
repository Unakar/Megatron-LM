"""Unit tests for HyperballAdam optimizer."""

import torch
import pytest
import sys
sys.path.append("/n/home07/nabreu/megatron/Megatron-LM")

try:
    from emerging_optimizers.scalar_optimizers.hyperball_adam import HyperballAdam
    HAVE_HYPERBALL_ADAM = True
except ImportError:
    HAVE_HYPERBALL_ADAM = False


@pytest.mark.skipif(not HAVE_HYPERBALL_ADAM, reason="HyperballAdam not available")
class TestHyperballAdam:
    """Test HyperballAdam optimizer functionality."""

    def test_basic_step(self):
        """Test that a basic optimization step changes weights and preserves Frobenius norm."""
        torch.manual_seed(42)

        W = torch.randn(128, 64, dtype=torch.float32)
        R_init = torch.norm(W, p="fro").item()

        opt = HyperballAdam([W], lr=0.01, betas=(0.9, 0.999), eps=1e-8)

        W.grad = torch.randn_like(W)
        W_before = W.clone()

        opt.step()

        # Weights should have changed
        assert not torch.allclose(W, W_before), "Weights did not change after step"

        # Frobenius norm should be preserved (equal to initial norm)
        R_after = torch.norm(W, p="fro").item()
        print(f"R_init={R_init:.4f}, R_after={R_after:.4f}, ratio={R_after/R_init:.6f}")
        assert abs(R_after - R_init) / R_init < 1e-4, (
            f"Frobenius norm {R_after:.4f} not close to initial {R_init:.4f}"
        )

    def test_frobenius_norm_maintained_over_steps(self):
        """Test that Frobenius norm is maintained over multiple optimization steps."""
        torch.manual_seed(42)

        W = torch.randn(256, 128, dtype=torch.float32)
        R_init = torch.norm(W, p="fro").item()

        opt = HyperballAdam([W], lr=0.02, betas=(0.9, 0.999), eps=1e-8)

        num_steps = 20
        for i in range(num_steps):
            W.grad = torch.randn_like(W) * 0.1
            opt.step()

            R = torch.norm(W, p="fro").item()
            ratio = R / R_init
            print(f"Step {i+1}: R={R:.4f}, R_init={R_init:.4f}, ratio={ratio:.6f}")

            assert abs(R - R_init) / R_init < 1e-4, (
                f"Step {i+1}: Frobenius norm {R:.4f} drifted from initial {R_init:.4f}"
            )

    def test_different_shapes(self):
        """Test HyperballAdam on various matrix shapes."""
        torch.manual_seed(42)

        shapes = [
            (512, 256),
            (256, 512),
            (1024, 1024),
            (2048, 512),
            (64, 64),
        ]

        for shape in shapes:
            W = torch.randn(*shape, dtype=torch.float32)
            R_init = torch.norm(W, p="fro").item()

            opt = HyperballAdam([W], lr=0.01)

            W.grad = torch.randn_like(W)
            opt.step()

            R_after = torch.norm(W, p="fro").item()
            ratio = R_after / R_init
            print(f"Shape {shape}: R_init={R_init:.4f}, R_after={R_after:.4f}, ratio={ratio:.6f}")

            assert abs(R_after - R_init) / R_init < 1e-4, (
                f"Shape {shape}: Frobenius norm {R_after:.4f} not close to {R_init:.4f}"
            )

    @pytest.mark.parametrize("bias_correction", [True, False])
    def test_bias_correction_toggle(self, bias_correction):
        """Test HyperballAdam with and without bias correction."""
        torch.manual_seed(42)

        W = torch.randn(128, 64, dtype=torch.float32)
        R_init = torch.norm(W, p="fro").item()

        opt = HyperballAdam([W], lr=0.01, bias_correction=bias_correction)

        for _ in range(5):
            W.grad = torch.randn_like(W) * 0.1
            opt.step()

        R_after = torch.norm(W, p="fro").item()
        print(f"bias_correction={bias_correction}: R_init={R_init:.4f}, R_after={R_after:.4f}")

        assert abs(R_after - R_init) / R_init < 1e-4

    def test_multiple_param_groups(self):
        """Test HyperballAdam with multiple parameters."""
        torch.manual_seed(42)

        W1 = torch.randn(128, 64, dtype=torch.float32)
        W2 = torch.randn(256, 128, dtype=torch.float32)
        R1_init = torch.norm(W1, p="fro").item()
        R2_init = torch.norm(W2, p="fro").item()

        opt = HyperballAdam([W1, W2], lr=0.01)

        for _ in range(5):
            W1.grad = torch.randn_like(W1) * 0.1
            W2.grad = torch.randn_like(W2) * 0.1
            opt.step()

        R1_after = torch.norm(W1, p="fro").item()
        R2_after = torch.norm(W2, p="fro").item()

        print(f"W1: R_init={R1_init:.4f}, R_after={R1_after:.4f}")
        print(f"W2: R_init={R2_init:.4f}, R_after={R2_after:.4f}")

        assert abs(R1_after - R1_init) / R1_init < 1e-4
        assert abs(R2_after - R2_init) / R2_init < 1e-4

    def test_zero_grad_no_crash(self):
        """Test that zero gradients don't cause division by zero."""
        torch.manual_seed(42)

        W = torch.randn(64, 32, dtype=torch.float32)

        opt = HyperballAdam([W], lr=0.01)

        # Zero gradient
        W.grad = torch.zeros_like(W)
        opt.step()  # Should not raise

        R_init = torch.norm(W, p="fro").item()
        # Norm should still be reasonable (unchanged from init since update direction is zero)
        assert R_init > 0, "Frobenius norm should be positive"


if __name__ == "__main__":
    # Run tests directly
    if HAVE_HYPERBALL_ADAM:
        test = TestHyperballAdam()
        print("Testing basic step...")
        test.test_basic_step()
        print("\nTesting Frobenius norm maintenance...")
        test.test_frobenius_norm_maintained_over_steps()
        print("\nTesting different shapes...")
        test.test_different_shapes()
        print("\nTesting bias correction toggle...")
        test.test_bias_correction_toggle(True)
        test.test_bias_correction_toggle(False)
        print("\nTesting multiple param groups...")
        test.test_multiple_param_groups()
        print("\nTesting zero grad...")
        test.test_zero_grad_no_crash()
        print("\n✓ All tests passed!")
    else:
        print("HyperballAdam not available, skipping tests")
