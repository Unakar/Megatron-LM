# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Spectral Ball Optimizer implementation."""

from typing import Any, Callable, Optional, Tuple

import torch
from absl import logging
from torch.optim.optimizer import ParamsT

from emerging_optimizers.mixin import WeightDecayT
from emerging_optimizers.orthogonalized_optimizers.orthogonalized_optimizer import (
    OrthogonalizedOptimizer,
    _args_doc,
)
from .spectral_ball_utils import compute_spectral_ball_update, compute_target_radius, get_spectral_ball_scale_factor


class SpectralBall(OrthogonalizedOptimizer):
    """Spectral Ball Optimizer with constrained optimization on spectral sphere.

    This optimizer constrains weight matrices to lie on a spectral sphere of fixed radius R,
    where ||W||_2 = R. The optimization proceeds by:

    1. Power iteration to compute spectral norm σ and top singular vectors (u, v)
    2. Retraction to spectral sphere: W ← (R/σ) * W
    3. Form Θ = u @ v^T
    4. Solve for Lagrange multiplier λ : <Θ, msign(M + λΘ)> = 0
    5. Compute update direction: Φ = msign(M + λΘ)
    6. Update: W ← W - lr * Φ

    The key insight is that the retraction step at the end of iteration t is equivalent to
    the retraction at the beginning of iteration t+1. This allows us to unify the power
    iteration for both retraction and Theta computation in a single efficient step.

    References:
        - Spectral MuP: Spectral Control of Feature Learning
        - Modular Duality in Deep Learning. arXiv:2410.21265 (2024).

    Warning:
        - This optimizer requires that all parameters passed in are 2D.
        - It should not be used for the embedding layer, the final fully connected layer,
          or any 1-D parameters; those should all be optimized by a standard method (e.g., AdamW).

    Note:
        The msign function always uses Polar-Express coefficients for optimal convergence.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum_beta: float = 0.9,
        weight_decay: float = 0.01,
        *,
        use_nesterov: bool = True,
        weight_decay_method: WeightDecayT = "decoupled",
        fp32_matmul_prec: str = "medium",
        power_iteration_steps: int = 10,
        msign_steps: int = 5,
        solver: str = "bisection",
        solver_tolerance_f: float = 1e-8,
        solver_max_iterations: int = 100,
        radius_mode: str = "spectral_mup",
        scale_mode: str = "align_adamw_rms",
        retract_mode: str = "hard",
        retract_alpha: float = 0.05,
        # QKV / TP support (optional)
        split_qkv: bool = False,
        is_qkv_fn: Optional[Callable[[torch.Tensor], bool]] = None,
        qkv_split_shapes: Optional[Tuple[int, int, int]] = None,
        pg_collection: Any | None = None,
        tp_mode: str = "duplicated",
    ) -> None:
        if power_iteration_steps < 1:
            raise ValueError(f"power_iteration_steps must be at least 1, got {power_iteration_steps}")
        if msign_steps < 1:
            raise ValueError(f"msign_steps must be at least 1, got {msign_steps}")
        if solver not in ("bisection"):
            raise ValueError(f"Invalid solver: {solver}, must be one of:  bisection")
        if radius_mode not in ("spectral_mup", "identity", "initialize"):
            raise ValueError(f"Invalid radius_mode: {radius_mode}, must be one of: spectral_mup, identity, initialize")
        if retract_mode not in ("hard", "dynamic"):
            raise ValueError(f"Invalid retract_mode: {retract_mode}, must be one of: hard, dynamic")

        # Store spectral ball specific parameters
        self.power_iteration_steps = power_iteration_steps
        self.msign_steps = msign_steps
        self.solver = solver
        self.solver_tolerance_f = solver_tolerance_f
        self.solver_max_iterations = solver_max_iterations
        self.radius_mode = radius_mode
        self.scale_mode = scale_mode
        self.retract_mode = retract_mode
        self.retract_alpha = retract_alpha
        self.retract_bias_dict = {}  # For logging retract bias (only in dynamic mode)
        # QKV / TP
        self.split_qkv = split_qkv
        self.is_qkv_fn = is_qkv_fn
        self.qkv_split_shapes = qkv_split_shapes
        self.pg_collection = pg_collection
        self.tp_mode = tp_mode

        # Placeholder for scaled_orthogonalize_fn
        # SpectralBall uses custom orthogonalize() method instead
        def scaled_orthogonalize_fn(grad: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError(
                "SpectralBall uses custom orthogonalize() method. "
                "scaled_orthogonalize_fn should not be called directly."
            )

        super().__init__(
            params,
            lr,
            momentum_beta,
            use_nesterov=use_nesterov,
            weight_decay=weight_decay,
            weight_decay_method=weight_decay_method,
            fp32_matmul_prec=fp32_matmul_prec,
            scaled_orthogonalize_fn=scaled_orthogonalize_fn,
            log_per_module_update_rms=False,  # Will be set later via config
        )

    def orthogonalize(self, p: torch.Tensor, grad: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Compute spectral ball update direction.

        This method overrides the base class orthogonalize() to implement the spectral ball
        constrained optimization. The input 'grad' is actually the momentum M (potentially
        with Nesterov momentum applied by the base class).

        The core algorithm:
        1. Power iteration: σ, u, v = power_iteration(W, steps)
        2. Retract W: W ← (R/σ) * W  [in-place modification]
        3. Form Θ: Θ = u @ v^T
        4. Solve: λ such that <Θ, msign(M + λΘ)> = 0
        5. Return: Φ = msign(M + λΘ)

        Args:
            p: Parameter tensor (current weight matrix W)
            grad: Momentum tensor M (after Nesterov if applicable)
            **kwargs: Additional parameters from param_group (unused)

        Returns:
            Update direction Φ to be applied as: W ← W - lr * Φ
        """
        # Clear retract bias dict at the start of each step
        # (will be repopulated by compute_spectral_ball_update calls)
        if not self.retract_bias_dict:
            self.retract_bias_dict.clear()
        # Compute target radius (no caching needed - it's a pure function of shape and mode)
        target_radius = compute_target_radius(
            shape=p.shape,
            radius_mode=self.radius_mode,
        )

        # Resolve TP group and partition dim if available
        tp_group = None
        partition_dim = None
        if self.pg_collection is not None:
            try:
                tp_group = (
                    self.pg_collection.expt_tp if getattr(p, "expert_tp", False) else self.pg_collection.tp
                )
            except Exception:
                tp_group = None
        if hasattr(p, "partition_dim"):
            partition_dim = getattr(p, "partition_dim")
            if partition_dim == -1:
                partition_dim = None

        # QKV splitting path
        if self.split_qkv and self.is_qkv_fn is not None and self.is_qkv_fn(p):
            assert self.qkv_split_shapes is not None, "qkv_split_shapes must be provided when split_qkv=True"
            out_dim, in_dim = p.shape
            split_sum = sum(self.qkv_split_shapes)
            assert (
                out_dim % split_sum == 0
            ), f"QKV split shapes {self.qkv_split_shapes} do not divide output dim {out_dim}"
            num_groups = out_dim // split_sum

            # reshape and split along the fused dimension (dim=1 after reshape)
            W_view = p.data.view(num_groups, split_sum, in_dim)
            M_view = grad.view(num_groups, split_sum, in_dim)
            W_q, W_k, W_v = torch.split(W_view, list(self.qkv_split_shapes), dim=1)
            M_q, M_k, M_v = torch.split(M_view, list(self.qkv_split_shapes), dim=1)

            # flatten per component to 2D matrices
            comps_W = [W_q.reshape(-1, in_dim), W_k.reshape(-1, in_dim), W_v.reshape(-1, in_dim)]
            comps_M = [M_q.reshape(-1, in_dim), M_k.reshape(-1, in_dim), M_v.reshape(-1, in_dim)]

            updates = []
            for idx, (Wi, Mi) in enumerate(zip(comps_W, comps_M)):
                # Compute per-component target radius (no caching needed)
                Ri = compute_target_radius(
                    shape=Wi.shape,
                    radius_mode=self.radius_mode,
                )

                ui, bias = compute_spectral_ball_update(
                    W=Wi,
                    M=Mi,
                    target_radius=Ri,
                    power_iteration_steps=self.power_iteration_steps,
                    msign_steps=self.msign_steps,
                    solver=self.solver,
                    solver_tolerance_f=self.solver_tolerance_f,
                    solver_max_iterations=self.solver_max_iterations,
                    tp_group=tp_group,
                    partition_dim=partition_dim,
                    tp_mode=self.tp_mode,
                    retract_mode=self.retract_mode,
                    retract_alpha=self.retract_alpha,
                )

                # Record bias for Q/K/V components (only if dynamic mode and bias != 0)
                if self.retract_mode == 'dynamic' and bias != 0.0:
                    param_name = getattr(p, 'param_name', None)
                    if param_name:
                        component_names = ['q', 'k', 'v']
                        self.retract_bias_dict[f"{param_name}.{component_names[idx]}"] = bias

                # Apply scale factor (mirroring Muon's approach)
                scale_factor = get_spectral_ball_scale_factor(Wi.shape[0], Wi.shape[1], mode=self.scale_mode)
                ui = ui * scale_factor

                # reshape back to [num_groups, part, in_dim]
                part_out = self.qkv_split_shapes[idx]
                updates.append(ui.view(num_groups, part_out, in_dim))

            # stitch back into fused shape
            U_q, U_k, U_v = updates
            update = torch.cat([U_q, U_k, U_v], dim=1).reshape(out_dim, in_dim)
            return update

        # Standard 2D matrix path
        update, bias = compute_spectral_ball_update(
            W=p.data,
            M=grad,
            target_radius=target_radius,
            power_iteration_steps=self.power_iteration_steps,
            msign_steps=self.msign_steps,
            solver=self.solver,
            solver_tolerance_f=self.solver_tolerance_f,
            solver_max_iterations=self.solver_max_iterations,
            tp_group=tp_group,
            partition_dim=partition_dim,
            tp_mode=self.tp_mode,
            retract_mode=self.retract_mode,
            retract_alpha=self.retract_alpha,
        )

        # Record bias (only if dynamic mode and bias != 0)
        if self.retract_mode == 'dynamic' and bias != 0.0:
            param_name = getattr(p, 'param_name', None)
            if param_name:
                self.retract_bias_dict[param_name] = bias

        # Apply scale factor (mirroring Muon's approach)
        scale_factor = get_spectral_ball_scale_factor(p.shape[0], p.shape[1], mode=self.scale_mode)
        update = update * scale_factor

        return update

    def get_retract_bias_dict(self):
        """Get retract bias dictionary for logging.

        Returns:
            Dictionary mapping module names to their retract bias values (-1 or +1),
            or None if retract_mode is 'hard' or dict is empty.
        """
        if self.retract_mode == 'hard' or not self.retract_bias_dict:
            return None
        return self.retract_bias_dict


SpectralBall.__doc__ = SpectralBall.__doc__.format(_args_doc=_args_doc)  # type: ignore[union-attr]
