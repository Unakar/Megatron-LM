# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Core SpectralBall optimizer implementation."""

import math
from typing import Optional

import torch
from torch.optim.optimizer import Optimizer, ParamsT

from .brent import solve_lambda_with_brent
from .msign import msign


class SpectralBallOptimizer(Optimizer):
    """Spectral Ball optimizer with Brent's method for Lagrange multiplier solving.

    This optimizer constrains weight matrices to lie on a spectral sphere of fixed radius R,
    i.e., ||W||_2 = R, using the optimization framework described in Spectral MuP.

    The update rule is:
    1. Compute SVD: U, S, Vh = svd(W) to get Theta = u_1 @ v_1^T
    2. Solve for λ using Brent: <Theta, msign(M + λ*Theta)> = 0 (where M is momentum)
    3. Compute Phi = msign(M + λ*Theta)
    4. Update: W_new = W - lr * Phi
    5. Retract to sphere: W = R/||W_new||_2 * W_new

    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate.
        momentum_beta: Momentum coefficient (default: 0.9).
        use_nesterov: Whether to use Nesterov momentum (default: True).
        weight_decay: Weight decay coefficient (default: 0.01).
        use_decoupled_weight_decay: Use decoupled weight decay (AdamW-style) (default: True).
        msign_steps: Number of Newton-Schulz iterations for msign (default: 5).
        brent_tolerance_f: Function tolerance for Brent solver (default: 1e-8).
        brent_tolerance_x: Variable tolerance for Brent solver (default: 1e-10).
        brent_max_iterations: Maximum iterations for Brent solver (default: 100).
        radius_mode: How to compute target radius R:
            - 'spectral_mup': R = sqrt(n_out / n_in) [default]
            - 'identity': R = 1.0
            - 'initialize': R = ||W||_2 at first step
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum_beta: float = 0.9,
        use_nesterov: bool = True,
        weight_decay: float = 0.01,
        use_decoupled_weight_decay: bool = True,
        msign_steps: int = 5,
        brent_tolerance_f: float = 1e-8,
        brent_tolerance_x: float = 1e-10,
        brent_max_iterations: int = 100,
        radius_mode: str = 'spectral_mup',
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

    def _compute_target_radius(self, param: torch.Tensor, radius_mode: str) -> float:
        """Compute target radius R for a parameter based on radius_mode.

        Args:
            param: Parameter tensor (shape: [n_out, n_in]).
            radius_mode: One of 'spectral_mup', 'identity', 'initialize'.

        Returns:
            Target radius R.
        """
        if radius_mode == 'spectral_mup':
            # R = sqrt(n_out / n_in)
            n_out, n_in = param.shape
            return math.sqrt(n_out / n_in)
        elif radius_mode == 'identity':
            return 1.0
        else:  # 'initialize'
            # Will be computed on first step
            return 0.0

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
                    # Compute and store target radius
                    state['target_radius'] = self._compute_target_radius(p, radius_mode)
                    if radius_mode == 'initialize':
                        # Initialize R from current spectral norm
                        with torch.no_grad():
                            current_norm = torch.linalg.matrix_norm(p.data, ord=2)
                            state['target_radius'] = float(current_norm.item())

                state['step'] += 1
                momentum_buffer = state['momentum_buffer']
                target_radius = state['target_radius']

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

                # Compute SVD to get top singular vectors
                # Use full_matrices=False for efficiency
                U, S, Vh = torch.linalg.svd(p.data, full_matrices=False)
                u1 = U[:, 0:1]  # Shape: [n_out, 1]
                v1 = Vh[0:1, :]  # Shape: [1, n_in]
                Theta = u1 @ v1  # Shape: [n_out, n_in]

                # Solve for lambda using Brent's method
                # Input G should be momentum M (not raw gradient!)
                result = solve_lambda_with_brent(
                    G=M,
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
                Z = M + lambda_value * Theta
                Phi = msign(Z, steps=msign_steps)

                # Apply decoupled weight decay (AdamW-style)
                if weight_decay != 0 and use_decoupled_weight_decay:
                    p.data.mul_(1 - lr * weight_decay)

                # Update: W_new = W - lr * Phi
                p.data.add_(Phi, alpha=-lr)

                # Retract to spectral sphere: W = R / ||W||_2 * W
                current_spectral_norm = torch.linalg.matrix_norm(p.data, ord=2)
                if current_spectral_norm > 0:
                    p.data.mul_(target_radius / current_spectral_norm)

        return loss
