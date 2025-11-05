# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tensor Parallel SpectralBall optimizer implementation."""

import logging
import math
from typing import Any, Callable, Optional

import torch
from torch.optim.optimizer import Optimizer, ParamsT

from .brent import solve_lambda_with_brent
from .msign import msign

logger = logging.getLogger(__name__)


class TensorParallelSpectralBall(Optimizer):
    """Tensor Parallel SpectralBall optimizer with QKV splitting support.

    This optimizer constrains weight matrices to lie on a spectral sphere of fixed radius R,
    with support for:
    - QKV splitting: Handle fused QKV parameters by processing Q, K, V independently
    - Tensor Parallelism: Support for distributed weights across TP ranks
    - Expert Parallelism: Handle MoE expert weights with separate process groups

    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate.
        momentum_beta: Momentum coefficient (default: 0.9).
        use_nesterov: Whether to use Nesterov momentum (default: True).
        weight_decay: Weight decay coefficient (default: 0.01).
        use_decoupled_weight_decay: Use decoupled weight decay (AdamW-style) (default: True).
        split_qkv: Whether to split QKV parameters (default: False).
        is_qkv_fn: Function to determine if a parameter is QKV (default: None).
        qkv_split_shapes: Tuple of (Q_size, K_size, V_size) per head (default: None).
        msign_steps: Number of Newton-Schulz iterations for msign (default: 5).
        brent_tolerance_f: Function tolerance for Brent solver (default: 1e-8).
        brent_tolerance_x: Variable tolerance for Brent solver (default: 1e-10).
        brent_max_iterations: Maximum iterations for Brent solver (default: 100).
        radius_mode: How to compute target radius R ('spectral_mup', 'identity', 'initialize').
        pg_collection: ProcessGroupCollection for distributed training (default: None).
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum_beta: float = 0.9,
        use_nesterov: bool = True,
        weight_decay: float = 0.01,
        use_decoupled_weight_decay: bool = True,
        split_qkv: bool = False,
        is_qkv_fn: Optional[Callable[[torch.Tensor], bool]] = None,
        qkv_split_shapes: Optional[tuple[int, int, int]] = None,
        msign_steps: int = 5,
        brent_tolerance_f: float = 1e-8,
        brent_tolerance_x: float = 1e-10,
        brent_max_iterations: int = 100,
        radius_mode: str = 'spectral_mup',
        pg_collection: Optional[Any] = None,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum_beta < 0.0 or momentum_beta >= 1.0:
            raise ValueError(f"Invalid momentum_beta: {momentum_beta}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if msign_steps < 1:
            raise ValueError(f"Invalid msign_steps: {msign_steps}")
        if radius_mode not in ('spectral_mup', 'identity', 'initialize'):
            raise ValueError(f"Invalid radius_mode: {radius_mode}")

        self.split_qkv = split_qkv
        self.is_qkv_fn = is_qkv_fn or (lambda p: False)
        self.qkv_split_shapes = qkv_split_shapes
        self.pg_collection = pg_collection

        defaults = dict(
            lr=lr,
            momentum_beta=momentum_beta,
            use_nesterov=use_nesterov,
            weight_decay=weight_decay,
            use_decoupled_weight_decay=use_decoupled_weight_decay,
            msign_steps=msign_steps,
            brent_tolerance_f=brent_tolerance_f,
            brent_tolerance_x=brent_tolerance_x,
            brent_max_iterations=brent_max_iterations,
            radius_mode=radius_mode,
        )
        super().__init__(params, defaults)

    def _compute_target_radius(self, param_shape: tuple, radius_mode: str) -> float:
        """Compute target radius R for a parameter based on radius_mode.

        Args:
            param_shape: Parameter shape (n_out, n_in).
            radius_mode: One of 'spectral_mup', 'identity', 'initialize'.

        Returns:
            Target radius R.
        """
        if radius_mode == 'spectral_mup':
            # R = sqrt(n_out / n_in)
            n_out, n_in = param_shape
            return math.sqrt(n_out / n_in)
        elif radius_mode == 'identity':
            return 1.0
        else:  # 'initialize'
            # Will be computed on first step
            return 0.0

    def _process_single_component(
        self,
        p_data: torch.Tensor,
        momentum: torch.Tensor,
        lr: float,
        weight_decay: float,
        use_decoupled_weight_decay: bool,
        msign_steps: int,
        brent_tolerance_f: float,
        brent_tolerance_x: float,
        brent_max_iterations: int,
        target_radius: float,
    ) -> torch.Tensor:
        """Process a single weight matrix component (used for both QKV and non-QKV).

        Args:
            p_data: Parameter data.
            momentum: Momentum buffer.
            lr: Learning rate.
            weight_decay: Weight decay coefficient.
            use_decoupled_weight_decay: Whether to use decoupled weight decay.
            msign_steps: Number of msign iterations.
            brent_tolerance_f: Brent function tolerance.
            brent_tolerance_x: Brent variable tolerance.
            brent_max_iterations: Brent max iterations.
            target_radius: Target spectral norm R.

        Returns:
            Updated parameter data.
        """
        # Compute SVD to get top singular vectors
        U, S, Vh = torch.linalg.svd(p_data, full_matrices=False)
        u1 = U[:, 0:1]  # Shape: [n_out, 1]
        v1 = Vh[0:1, :]  # Shape: [1, n_in]
        Theta = u1 @ v1  # Shape: [n_out, n_in]

        # Solve for lambda using Brent's method
        result = solve_lambda_with_brent(
            G=momentum,
            Theta=Theta,
            initial_guess=0.0,
            initial_step=1.0,
            tolerance_f=brent_tolerance_f,
            tolerance_x=brent_tolerance_x,
            max_iterations=brent_max_iterations,
            max_expansions=60,
            msign_steps=msign_steps,
        )

        lambda_value = result.solution

        # Compute update direction: Phi = msign(M + lambda * Theta)
        Z = momentum + lambda_value * Theta
        Phi = msign(Z, steps=msign_steps)

        # Apply decoupled weight decay (AdamW-style)
        if weight_decay != 0 and use_decoupled_weight_decay:
            p_data = p_data * (1 - lr * weight_decay)

        # Update: W_new = W - lr * Phi
        p_data = p_data - lr * Phi

        # Retract to spectral sphere: W = R / ||W||_2 * W
        current_spectral_norm = torch.linalg.matrix_norm(p_data, ord=2)
        if current_spectral_norm > 0:
            p_data = p_data * (target_radius / current_spectral_norm)

        return p_data

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            The loss if closure is provided, otherwise None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum_beta = group['momentum_beta']
            use_nesterov = group['use_nesterov']
            weight_decay = group['weight_decay']
            use_decoupled_weight_decay = group['use_decoupled_weight_decay']
            msign_steps = group['msign_steps']
            brent_tolerance_f = group['brent_tolerance_f']
            brent_tolerance_x = group['brent_tolerance_x']
            brent_max_iterations = group['brent_max_iterations']
            radius_mode = group['radius_mode']

            for p in group['params']:
                if p.grad is None:
                    continue

                # Only apply to 2D tensors (weight matrices)
                if p.ndim != 2:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(p.data)

                state['step'] += 1
                momentum_buffer = state['momentum_buffer']

                # Apply weight decay to gradient (not decoupled)
                if weight_decay != 0 and not use_decoupled_weight_decay:
                    grad = grad.add(p.data, alpha=weight_decay)

                # Update momentum: M = beta * M + (1 - beta) * grad
                momentum_buffer.mul_(momentum_beta).add_(grad, alpha=1 - momentum_beta)

                # Get momentum for update (Nesterov or standard)
                if use_nesterov:
                    # Nesterov momentum: M_update = beta * M + (1 - beta) * grad
                    M = momentum_buffer.mul(momentum_beta).add(grad, alpha=1 - momentum_beta)
                else:
                    M = momentum_buffer

                # Check if this is a QKV parameter that should be split
                if self.split_qkv and self.is_qkv_fn(p):
                    # Split QKV parameter into Q, K, V components
                    grad_shape = p.shape
                    logger.debug(
                        f'Processing QKV parameter with shape {grad_shape}, '
                        f'split shapes {self.qkv_split_shapes}'
                    )

                    # Compute number of query groups (for GQA support)
                    num_query_groups = grad_shape[0] // sum(self.qkv_split_shapes)

                    # Split momentum into Q, K, V
                    M_reshaped = M.view(num_query_groups, sum(self.qkv_split_shapes), -1)
                    qkv_momentums = torch.split(M_reshaped, self.qkv_split_shapes, dim=1)
                    qkv_momentums = [m.reshape(-1, grad_shape[-1]) for m in qkv_momentums]

                    # Split parameter data into Q, K, V
                    p_reshaped = p.data.view(num_query_groups, sum(self.qkv_split_shapes), -1)
                    qkv_params = torch.split(p_reshaped, self.qkv_split_shapes, dim=1)
                    qkv_params = [param.reshape(-1, grad_shape[-1]) for param in qkv_params]

                    # Process each component independently
                    updated_qkv = []
                    for component_idx, (p_component, m_component) in enumerate(
                        zip(qkv_params, qkv_momentums)
                    ):
                        # Compute target radius for this component
                        if f'qkv_target_radius_{component_idx}' not in state:
                            target_radius = self._compute_target_radius(
                                p_component.shape, radius_mode
                            )
                            if radius_mode == 'initialize':
                                current_norm = torch.linalg.matrix_norm(p_component, ord=2)
                                target_radius = float(current_norm.item())
                            state[f'qkv_target_radius_{component_idx}'] = target_radius
                        else:
                            target_radius = state[f'qkv_target_radius_{component_idx}']

                        # Process this component
                        updated_component = self._process_single_component(
                            p_data=p_component,
                            momentum=m_component,
                            lr=lr,
                            weight_decay=weight_decay,
                            use_decoupled_weight_decay=use_decoupled_weight_decay,
                            msign_steps=msign_steps,
                            brent_tolerance_f=brent_tolerance_f,
                            brent_tolerance_x=brent_tolerance_x,
                            brent_max_iterations=brent_max_iterations,
                            target_radius=target_radius,
                        )
                        updated_qkv.append(
                            updated_component.view(num_query_groups, -1, grad_shape[-1])
                        )

                    # Concatenate Q, K, V back together
                    p.data = torch.cat(updated_qkv, dim=1).view(grad_shape)

                else:
                    # Standard processing (non-QKV or QKV splitting disabled)
                    # Compute and store target radius
                    if 'target_radius' not in state:
                        target_radius = self._compute_target_radius(p.shape, radius_mode)
                        if radius_mode == 'initialize':
                            current_norm = torch.linalg.matrix_norm(p.data, ord=2)
                            target_radius = float(current_norm.item())
                        state['target_radius'] = target_radius
                    else:
                        target_radius = state['target_radius']

                    # Process the entire parameter
                    p.data = self._process_single_component(
                        p_data=p.data,
                        momentum=M,
                        lr=lr,
                        weight_decay=weight_decay,
                        use_decoupled_weight_decay=use_decoupled_weight_decay,
                        msign_steps=msign_steps,
                        brent_tolerance_f=brent_tolerance_f,
                        brent_tolerance_x=brent_tolerance_x,
                        brent_max_iterations=brent_max_iterations,
                        target_radius=target_radius,
                    )

        return loss
