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
from .spectral_ball_utils import compute_spectral_ball_update, compute_target_radius


class SpectralBall(OrthogonalizedOptimizer):
    """Spectral Ball Optimizer with constrained optimization on spectral sphere.

    This optimizer constrains weight matrices to lie on a spectral sphere of fixed radius R,
    where ||W||_2 = R. The optimization proceeds by:

    1. Power iteration to compute spectral norm σ and top singular vectors (u, v)
    2. Retraction to spectral sphere: W ← (R/σ) * W
    3. Form Θ = u @ v^T
    4. Solve for Lagrange multiplier λ using Brent's method: <Θ, msign(M + λΘ)> = 0
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

    Args:
        {_args_doc}
        power_iteration_steps: Number of power iteration steps to compute top singular vectors.
        msign_steps: Number of Newton-Schulz style iterations for msign computation.
        msign_coefficient_type: Coefficient type for msign ("polar_express", "quintic", etc.).
        brent_tolerance_f: Function tolerance for Brent solver convergence.
        brent_tolerance_x: Variable tolerance for Brent solver convergence.
        brent_max_iterations: Maximum iterations for Brent solver.
        radius_mode: How to compute target radius R:
            - "spectral_mup": R = sqrt(n_out / n_in) [default for μP-style scaling]
            - "identity": R = 1.0 [standard normalization]
            - "initialize": R = ||W||_2 at first step [preserve initial scale]
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
        msign_coefficient_type: str = "polar_express",
        brent_tolerance_f: float = 1e-8,
        brent_tolerance_x: float = 1e-10,
        brent_max_iterations: int = 100,
        radius_mode: str = "spectral_mup",
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
        if radius_mode not in ("spectral_mup", "identity", "initialize"):
            raise ValueError(f"Invalid radius_mode: {radius_mode}, must be one of: spectral_mup, identity, initialize")

        # Store spectral ball specific parameters
        self.power_iteration_steps = power_iteration_steps
        self.msign_steps = msign_steps
        self.msign_coefficient_type = msign_coefficient_type
        self.brent_tolerance_f = brent_tolerance_f
        self.brent_tolerance_x = brent_tolerance_x
        self.brent_max_iterations = brent_max_iterations
        self.radius_mode = radius_mode
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
        state = self.state[p]

        # Initialize target radius on first step
        if "target_radius" not in state:
            state["target_radius"] = compute_target_radius(
                shape=p.shape,
                radius_mode=self.radius_mode,
                current_weight=p.data if self.radius_mode == "initialize" else None,
            )
            logging.debug(
                f"Initialized target_radius={state['target_radius']:.6f} "
                f"for parameter shape {p.shape} with mode {self.radius_mode}"
            )

        target_radius = state["target_radius"]

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
                # per-component target radius (cache if needed)
                key = f"qkv_target_radius_{idx}"
                if key not in state:
                    state[key] = compute_target_radius(
                        shape=Wi.shape, radius_mode=self.radius_mode,
                        current_weight=Wi if self.radius_mode == "initialize" else None,
                    )
                Ri = state[key]

                ui = compute_spectral_ball_update(
                    W=Wi,
                    M=Mi,
                    target_radius=Ri,
                    power_iteration_steps=self.power_iteration_steps,
                    msign_steps=self.msign_steps,
                    msign_coefficient_type=self.msign_coefficient_type,
                    brent_tolerance_f=self.brent_tolerance_f,
                    brent_tolerance_x=self.brent_tolerance_x,
                    brent_max_iterations=self.brent_max_iterations,
                    tp_group=tp_group,
                    partition_dim=partition_dim,
                    tp_mode=self.tp_mode,
                )
                # reshape back to [num_groups, part, in_dim]
                part_out = self.qkv_split_shapes[idx]
                updates.append(ui.view(num_groups, part_out, in_dim))

            # stitch back into fused shape
            U_q, U_k, U_v = updates
            update = torch.cat([U_q, U_k, U_v], dim=1).reshape(out_dim, in_dim)
            return update

        # Standard 2D matrix path
        update = compute_spectral_ball_update(
            W=p.data,
            M=grad,
            target_radius=target_radius,
            power_iteration_steps=self.power_iteration_steps,
            msign_steps=self.msign_steps,
            msign_coefficient_type=self.msign_coefficient_type,
            brent_tolerance_f=self.brent_tolerance_f,
            brent_tolerance_x=self.brent_tolerance_x,
            brent_max_iterations=self.brent_max_iterations,
            tp_group=tp_group,
            partition_dim=partition_dim,
            tp_mode=self.tp_mode,
        )

        return update


SpectralBall.__doc__ = SpectralBall.__doc__.format(_args_doc=_args_doc)  # type: ignore[union-attr]
