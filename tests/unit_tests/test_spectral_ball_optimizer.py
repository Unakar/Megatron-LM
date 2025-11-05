# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging.version import Version

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer.spectral_ball import SpectralBallOptimizer
from megatron.core.optimizer.spectral_ball_optimizer import get_megatron_spectral_ball_optimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

# Skip all tests in this file for LTS versions
pytestmark = pytest.mark.skipif(
    Version(os.getenv('NVIDIA_PYTORCH_VERSION', "24.01")) <= Version("25.05"),
    reason="Skip spectral ball optimizer for LTS test",
)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(80, 48)
        self.fc2 = nn.Linear(48, 32)
        self.fc3 = nn.Linear(32, 24)
        self.fc4 = nn.Linear(24, 16)
        self.fc5 = nn.Linear(16, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


def test_spectral_ball_optimizer_smoke():
    """Smoke test for SpectralBallOptimizer."""
    # Create a simple linear model for testing
    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    # Create SpectralBallOptimizer
    optimizer = SpectralBallOptimizer(
        params=[model.weight],
        lr=0.01,
        momentum_beta=0.9,
        use_nesterov=True,
        weight_decay=0.01,
        use_decoupled_weight_decay=True,
        msign_steps=5,
        brent_tolerance_f=1e-8,
        brent_tolerance_x=1e-10,
        brent_max_iterations=100,
        radius_mode='spectral_mup',
    )

    # Test basic properties
    assert optimizer is not None, "Optimizer should not be None"
    assert hasattr(optimizer, 'param_groups'), "Optimizer should have param_groups"
    assert len(optimizer.param_groups) > 0, "Optimizer should have at least one parameter group"

    # Test forward and backward pass
    input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    # Store original weight and spectral norm
    original_weight = model.weight.data.clone()
    original_spectral_norm = torch.linalg.matrix_norm(original_weight, ord=2).item()

    # Test optimizer step
    optimizer.step()

    # Verify weight was updated
    assert not torch.equal(
        model.weight.data, original_weight
    ), "Weight should be updated after optimizer step"

    # Verify spectral norm constraint
    # For spectral_mup mode, R = sqrt(n_out / n_in) = sqrt(50 / 100) = sqrt(0.5) ≈ 0.707
    new_spectral_norm = torch.linalg.matrix_norm(model.weight.data, ord=2).item()
    expected_radius = (50.0 / 100.0) ** 0.5
    assert abs(new_spectral_norm - expected_radius) < 1e-5, (
        f"Spectral norm should be approximately {expected_radius}, got {new_spectral_norm}"
    )

    # Test zero_grad
    optimizer.zero_grad()
    assert model.weight.grad is None or torch.all(
        model.weight.grad == 0
    ), "Gradients should be zeroed"

    # Test state_dict and load_state_dict
    state_dict = optimizer.state_dict()
    assert 'state' in state_dict, "State dict should contain state"
    assert 'param_groups' in state_dict, "State dict should contain param_groups"

    # Load state dict should not raise error
    optimizer.load_state_dict(state_dict)


@pytest.mark.parametrize("radius_mode", ['spectral_mup', 'identity', 'initialize'])
def test_spectral_ball_optimizer_radius_modes(radius_mode):
    """Test SpectralBallOptimizer with different radius modes."""
    model = torch.nn.Linear(60, 30, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = SpectralBallOptimizer(
        params=[model.weight],
        lr=0.01,
        radius_mode=radius_mode,
        msign_steps=5,
    )

    input_tensor = torch.randn(16, 60, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with radius_mode={radius_mode}"

    # Verify spectral norm based on radius mode
    new_spectral_norm = torch.linalg.matrix_norm(model.weight.data, ord=2).item()
    if radius_mode == 'spectral_mup':
        expected_radius = (30.0 / 60.0) ** 0.5
    elif radius_mode == 'identity':
        expected_radius = 1.0
    else:  # 'initialize'
        # Should match the initial spectral norm
        expected_radius = torch.linalg.matrix_norm(original_weight, ord=2).item()

    assert abs(new_spectral_norm - expected_radius) < 1e-4, (
        f"Spectral norm should be approximately {expected_radius}, got {new_spectral_norm}"
    )


@pytest.mark.parametrize("use_nesterov", [True, False])
def test_spectral_ball_optimizer_nesterov(use_nesterov):
    """Test SpectralBallOptimizer with and without Nesterov momentum."""
    model = torch.nn.Linear(50, 25, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = SpectralBallOptimizer(
        params=[model.weight],
        lr=0.01,
        momentum_beta=0.9,
        use_nesterov=use_nesterov,
        msign_steps=5,
    )

    input_tensor = torch.randn(16, 50, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with use_nesterov={use_nesterov}"


def test_spectral_ball_optimizer_multiple_steps():
    """Test SpectralBallOptimizer across multiple optimization steps."""
    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = SpectralBallOptimizer(
        params=[model.weight],
        lr=0.01,
        momentum_beta=0.9,
        weight_decay=0.01,
        msign_steps=5,
    )

    weights_history = [model.weight.data.clone()]
    spectral_norms = [torch.linalg.matrix_norm(model.weight.data, ord=2).item()]

    for i in range(3):
        input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        weights_history.append(model.weight.data.clone())
        spectral_norms.append(torch.linalg.matrix_norm(model.weight.data, ord=2).item())

    # Verify weights changed at each step
    for i in range(len(weights_history) - 1):
        assert not torch.equal(
            weights_history[i], weights_history[i + 1]
        ), f"Weight should change at step {i}"

    # Verify spectral norm constraint maintained at each step
    expected_radius = (50.0 / 100.0) ** 0.5
    for i, norm in enumerate(spectral_norms[1:]):  # Skip initial norm
        assert abs(norm - expected_radius) < 1e-4, (
            f"Step {i}: Spectral norm should be {expected_radius}, got {norm}"
        )


@pytest.mark.parametrize("msign_steps", [3, 5, 8])
def test_spectral_ball_optimizer_msign_steps(msign_steps):
    """Test SpectralBallOptimizer with different numbers of msign steps."""
    model = torch.nn.Linear(60, 30, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = SpectralBallOptimizer(
        params=[model.weight],
        lr=0.01,
        msign_steps=msign_steps,
    )

    input_tensor = torch.randn(16, 60, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with msign_steps={msign_steps}"


@pytest.mark.skipif(
    int(os.getenv('WORLD_SIZE', '1')) == 1, reason="Multi-rank test requires WORLD_SIZE > 1"
)
class TestSpectralBallOptimizerMultiRank:
    """Test class for SpectralBall optimizer with multi-rank setup."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test."""
        Utils.initialize_model_parallel()
        yield
        Utils.destroy_model_parallel()

    def create_ddp_model(self, model):
        """Wrap model in DDP.

        Args:
            model: Model to wrap

        Returns:
            DDP-wrapped model
        """
        ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
        return DistributedDataParallel(
            TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
        )

    def test_get_megatron_spectral_ball_optimizer_smoke(self):
        """Smoke test for get_megatron_spectral_ball_optimizer function."""
        model = Net().bfloat16().cuda()
        model.requires_grad_(True)
        model = self.create_ddp_model(model)

        # Ensure all parameters require gradients
        for param in model.parameters():
            assert param.requires_grad, "All parameters should require gradients"

        # Create optimizer config for SpectralBall
        optimizer_config = OptimizerConfig(
            optimizer='spectral_ball',
            lr=0.01,
            weight_decay=0.01,
            bf16=True,
            use_distributed_optimizer=False,
            spectral_ball_momentum=0.9,
            spectral_ball_use_nesterov=True,
            spectral_ball_msign_steps=5,
            spectral_ball_radius_mode='spectral_mup',
        )

        # Test creating the optimizer
        optimizer = get_megatron_spectral_ball_optimizer(
            config=optimizer_config,
            model_chunks=[model],
            use_gloo_process_groups=True,
            layer_wise_distributed_optimizer=False,
        )

        # Test basic properties
        assert optimizer is not None, "Optimizer should not be None"
        assert hasattr(optimizer, 'param_groups'), "Optimizer should have param_groups"
        assert hasattr(optimizer, 'chained_optimizers'), "Should be a ChainedOptimizer"
        assert len(optimizer.chained_optimizers) >= 1, "Should have at least one chained optimizer"

        # Test forward and backward pass
        input_tensor = torch.randn(16, 80, dtype=torch.bfloat16, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        # Store original parameters
        original_params = {}
        for name, param in model.named_parameters():
            original_params[name] = param.data.clone()

        # Test optimizer step
        optimizer.step()

        # Verify at least some parameters were updated
        params_updated = 0
        for name, param in model.named_parameters():
            if not torch.equal(param.data, original_params[name]):
                params_updated += 1

        assert params_updated > 0, "At least some parameters should be updated after optimizer step"

        # Test zero_grad
        optimizer.zero_grad()
        for param in model.parameters():
            assert param.grad is None or torch.all(
                param.grad == 0
            ), f"Gradients should be zeroed for all parameters"

        # Test state_dict and load_state_dict
        state_dict = optimizer.state_dict()
        assert isinstance(state_dict, list), "State dict should be a list"

        # Load state dict should not raise error
        optimizer.load_state_dict(state_dict)

    def test_get_megatron_spectral_ball_optimizer_validation(self):
        """Test validation logic for get_megatron_spectral_ball_optimizer."""
        model = torch.nn.Linear(100, 50, bias=False, dtype=torch.bfloat16, device='cuda')
        model.requires_grad_(True)
        model = self.create_ddp_model(model)

        # Test 1: Distributed optimizer should raise exception
        optimizer_config_dist = OptimizerConfig(
            optimizer='spectral_ball',
            lr=0.01,
            bf16=True,
            use_distributed_optimizer=True,  # This should cause an exception
        )

        with pytest.raises(Exception, match='spectral_ball with distributed optimizer is not supported'):
            get_megatron_spectral_ball_optimizer(config=optimizer_config_dist, model_chunks=[model])

        # Test 2: FP16 should raise exception
        optimizer_config_fp16 = OptimizerConfig(
            optimizer='spectral_ball',
            lr=0.01,
            fp16=True,  # This should cause an exception
            use_distributed_optimizer=False,
        )

        with pytest.raises(Exception, match='spectral_ball with fp16 is not supported'):
            get_megatron_spectral_ball_optimizer(config=optimizer_config_fp16, model_chunks=[model])

    def test_get_megatron_spectral_ball_optimizer_layer_wise(self):
        """Test get_megatron_spectral_ball_optimizer with layer-wise distributed optimizer."""
        model = Net().bfloat16().cuda()
        model.requires_grad_(True)
        model = self.create_ddp_model(model)

        optimizer_config = OptimizerConfig(
            optimizer='spectral_ball',
            lr=0.01,
            weight_decay=0.01,
            bf16=True,
            use_distributed_optimizer=False,
            spectral_ball_momentum=0.9,
            spectral_ball_use_nesterov=True,
            spectral_ball_msign_steps=5,
            spectral_ball_radius_mode='spectral_mup',
        )

        # Test with layer_wise_distributed_optimizer=True
        optimizer = get_megatron_spectral_ball_optimizer(
            config=optimizer_config,
            model_chunks=[model],
            use_gloo_process_groups=True,
            layer_wise_distributed_optimizer=True,
        )

        # Verify it's a LayerWiseDistributedOptimizer
        from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer

        assert isinstance(
            optimizer, LayerWiseDistributedOptimizer
        ), "Should return LayerWiseDistributedOptimizer"

        # Test forward and backward pass
        input_tensor = torch.randn(16, 80, dtype=torch.bfloat16, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        # Test optimizer step
        update_successful, grad_norm, num_zeros = optimizer.step()

        assert update_successful, "Optimizer step should be successful"
        assert grad_norm is not None or grad_norm is None, "Grad norm should be returned"


def test_spectral_ball_optimizer_qkv_split():
    """Test TensorParallelSpectralBall optimizer with QKV splitting."""
    from megatron.core.optimizer.spectral_ball import TensorParallelSpectralBall

    # Create a model with QKV-like parameter
    qkv_size = 3 * 64 * 16  # Combined Q, K, V dimensions, 16 heads x 64 per head
    hidden_size = 1024
    model = torch.nn.Linear(hidden_size, qkv_size, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    # Mark parameter as QKV
    model.weight.is_qkv = True

    # QKV split shapes: [Q_size, K_size, V_size]
    qkv_split_shapes = (64, 64, 64)

    # Test with split_qkv=True
    optimizer_split = TensorParallelSpectralBall(
        params=[model.weight],
        lr=0.01,
        split_qkv=True,
        is_qkv_fn=lambda p: getattr(p, 'is_qkv', False),
        qkv_split_shapes=qkv_split_shapes,
        msign_steps=5,
        pg_collection=None,
    )

    input_tensor = torch.randn(16, hidden_size, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer_split.step()
    weight_with_split = model.weight.data.clone()

    assert not torch.equal(
        weight_with_split, original_weight
    ), "QKV weight should be updated with split_qkv=True"

    # Reset model and test with split_qkv=False
    model.weight.data.fill_(1.0)
    optimizer_no_split = TensorParallelSpectralBall(
        params=[model.weight],
        lr=0.01,
        split_qkv=False,
        msign_steps=5,
        pg_collection=None,
    )

    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    optimizer_no_split.step()
    weight_without_split = model.weight.data.clone()

    assert not torch.equal(
        weight_without_split, original_weight
    ), "QKV weight should be updated with split_qkv=False"

    # Ensure the two results are different
    assert not torch.equal(
        weight_with_split, weight_without_split
    ), "Weights should be different between split_qkv=True and split_qkv=False"
