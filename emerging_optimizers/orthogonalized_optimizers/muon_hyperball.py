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

"""MuonHyperball Optimizer: Muon orthogonalization with Frobenius-norm sphere projection.

MuonHyperball combines:
- Muon's msign (Newton-Schulz) orthogonalization for the update direction
- Frobenius-norm sphere retraction (instead of spectral norm)

Algorithm:
1. Compute Frobenius norm: f = ||W||_F
2. Retract W to Frobenius sphere: W ← (R/f) * W
3. Orthogonalize: Φ = msign(M)
4. Update: W ← W - lr * scale * Φ

Key difference from MuonBall:
- MuonBall uses spectral norm (via power iteration) for retraction
- MuonHyperball uses Frobenius norm (direct computation) — cheaper, no power iteration needed
"""

from typing import Any, Callable, Optional, Tuple

import torch
from absl import logging
from torch.optim.optimizer import ParamsT

from emerging_optimizers.mixin import WeightDecayT
from emerging_optimizers.orthogonalized_optimizers.orthogonalized_optimizer import (
    OrthogonalizedOptimizer,
    _args_doc,
)
from .spectral_ball_utils import (
    msign,
)


def _compute_frobenius_target_radius(
    shape: tuple,
    radius_mode: str,
) -> float:
    """Compute target Frobenius radius R.

    Args:
        shape: (n_out, n_in) shape of the weight matrix.
        radius_mode: How to compute R:
            - "frobenius_mup": R = sqrt(n_out) (analogous to spectral_mup but for Frobenius)
            - "identity": R = 1.0
            - "initialize": R will be captured from the initial weight at first step

    Returns:
        Target Frobenius radius, or -1.0 for "initialize" mode (deferred to first step).
    """
    if radius_mode == "frobenius_mup":
        n_out, n_in = shape
        # For a random Gaussian matrix, E[||W||_F] ≈ sqrt(n_out * n_in) * sigma
        # With kaiming init (sigma=1/sqrt(n_in)), E[||W||_F] ≈ sqrt(n_out)
        return float(n_out ** 0.5)
    elif radius_mode == "identity":
        return 1.0
    elif radius_mode == "initialize":
        return -1.0  # Will be set from ||W_0||_F on first step
    else:
        raise ValueError(
            f"Invalid radius_mode: {radius_mode}. "
            f"Must be 'frobenius_mup', 'identity', or 'initialize'."
        )


@torch.no_grad()
def _apply_frobenius_retract(
    W: torch.Tensor,
    target_radius: float,
) -> float:
    """Apply hard retraction to Frobenius-norm sphere.

    Rescales W in-place so that ||W||_F = target_radius.

    Args:
        W: Weight matrix (modified in-place).
        target_radius: Target Frobenius norm R.

    Returns:
        The Frobenius norm of W before retraction.
    """
    frob_norm = torch.norm(W.float(), p="fro").item()
    if frob_norm > 0:
        W.mul_(target_radius / frob_norm)
    return frob_norm


def compute_muon_hyperball_update(
    W: torch.Tensor,
    M: torch.Tensor,
    target_radius: float,
    msign_steps: int,
    *,
    tp_group: torch.distributed.ProcessGroup | None = None,
    partition_dim: int | None = None,
) -> Tuple[torch.Tensor, float]:
    """Compute MuonHyperball update direction: R * Normalize(msign(M)).

    Implements the update direction for:
        W_{t+1} = R * Normalize(W_t - lr * R * Normalize(u_t))
    where u_t = msign(M) is the standard Muon update.

    This function computes the update direction Phi = R * u / ||u||_F.
    The caller is responsible for:
    1. Applying the update: W <- W - lr * Phi
    2. Projecting to the Frobenius sphere: W <- R * W / ||W||_F

    Args:
        W: Current weight matrix (NOT modified).
        M: Momentum tensor.
        target_radius: Target Frobenius norm R.
        msign_steps: Number of Newton-Schulz iterations for msign.
        tp_group: Tensor parallel process group (None for single-rank).
        partition_dim: Dimension along which tensors are partitioned.

    Returns:
        Tuple of (Phi, frob_norm) where:
        - Phi: Update direction R * Normalize(msign(M))
        - frob_norm: Current Frobenius norm of W (for logging)
    """
    # Handle TP if enabled
    from .spectral_ball_utils import _tp_world_and_rank, _tp_gather_along_dim, _tp_split_along_dim

    ws, _ = _tp_world_and_rank(tp_group)
    tp_enabled = tp_group is not None and partition_dim is not None and ws > 1

    if tp_enabled:
        W_full = _tp_gather_along_dim(W, tp_group, partition_dim)
        M_full = _tp_gather_along_dim(M, tp_group, partition_dim)
        W_work = W_full
        M_work = M_full
    else:
        W_work = W
        M_work = M

    # Record Frobenius norm of W for logging
    frob_norm = torch.norm(W_work.float(), p="fro").item()

    # Standard Muon update: u = msign(M)
    M_fp32 = M_work.to(torch.float32)
    u = msign(M_fp32, steps=msign_steps)

    # Normalize and scale by R: Phi = R * u / ||u||_F
    u_frob = torch.norm(u, p="fro").clamp_min(1e-8)
    Phi = target_radius * (u / u_frob)

    # Handle TP: split result
    if tp_enabled:
        Phi_local = _tp_split_along_dim(Phi, tp_group, partition_dim)
        return Phi_local, frob_norm
    else:
        return Phi, frob_norm


class MuonHyperball(OrthogonalizedOptimizer):
    """MuonHyperball Optimizer: Muon with Frobenius-norm sphere constraint.

    MuonHyperball uses Muon's msign orthogonalization for the update direction,
    but constrains weight matrices to a Frobenius-norm sphere instead of a
    spectral-norm sphere (as in MuonBall).

    The algorithm:
    1. Compute Frobenius norm f = ||W||_F
    2. Retraction to Frobenius sphere: W ← (R/f) * W
    3. Orthogonalize momentum: Φ = msign(M)
    4. Update: W ← W - lr * scale * Φ

    Comparison to MuonBall:
    - MuonBall: Retracts using spectral norm via power iteration (expensive)
    - MuonHyperball: Retracts using Frobenius norm (cheap, direct computation)

    Comparison to HyperballAdam:
    - HyperballAdam: Uses Adam for the update direction, Frobenius retraction
    - MuonHyperball: Uses Muon/msign for the update direction, Frobenius retraction

    Warning:
        - This optimizer requires that all parameters passed in are 2D.
        - It should not be used for the embedding layer, the final fully connected layer,
          or any 1-D parameters; those should all be optimized by a standard method (e.g., AdamW).

    Args:
        {_args_doc}
        msign_steps: Number of Newton-Schulz iterations for msign (uses Polar-Express).
        radius_mode: Target radius mode ("frobenius_mup", "identity", "initialize").
        scale_mode: Scale factor mode for updates ("align_adamw_rms", "shape_scaling", "spectral_mup").
        split_qkv: Whether to split QKV parameters and process Q/K/V independently.
        is_qkv_fn: Function to identify QKV parameters.
        qkv_split_shapes: Tuple of (q_dim, k_dim, v_dim) per query group.
        qkv_split_mode: QKV split mode ("component", "group", or "head").
        split_fc1: Whether to split FC1 (gate and up) for gated linear units.
        is_fc1_fn: Function to identify FC1 parameters.
        fc1_split_shapes: Tuple of (gate_dim, up_dim).
        split_moe_experts: Whether to split GroupedMLP experts and process independently.
        is_grouped_moe_fn: Function to identify GroupedMLP parameters (weight1/weight2).
        pg_collection: ProcessGroupCollection for tensor parallel support.
        tp_mode: Tensor parallel mode ("duplicated", "blockwise", or "distributed").
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
        msign_steps: int = 5,
        radius_mode: str = "initialize",
        scale_mode: str = "align_adamw_rms",
        # QKV / TP support (optional)
        split_qkv: bool = False,
        is_qkv_fn: Optional[Callable[[torch.Tensor], bool]] = None,
        qkv_split_shapes: Optional[Tuple[int, int, int]] = None,
        qkv_split_mode: str = "component",
        # FC1 split support for gated linear units (SwiGLU)
        split_fc1: bool = False,
        is_fc1_fn: Optional[Callable[[torch.Tensor], bool]] = None,
        fc1_split_shapes: Optional[Tuple[int, int]] = None,
        # MoE expert split support for GroupedMLP
        split_moe_experts: bool = False,
        is_grouped_moe_fn: Optional[Callable[[torch.Tensor], bool]] = None,
        pg_collection: Any | None = None,
        tp_mode: str = "duplicated",
    ) -> None:
        if msign_steps < 1:
            raise ValueError(f"msign_steps must be at least 1, got {msign_steps}")
        if radius_mode not in ("frobenius_mup", "identity", "initialize"):
            raise ValueError(
                f"Invalid radius_mode: {radius_mode}, "
                f"must be one of: frobenius_mup, identity, initialize"
            )
        if qkv_split_mode not in ("component", "group", "head"):
            raise ValueError(
                f"Invalid qkv_split_mode: {qkv_split_mode}, "
                f"must be one of: component, group, head"
            )

        # Store MuonHyperball specific parameters
        self.msign_steps = msign_steps
        self.radius_mode = radius_mode
        self.scale_mode = scale_mode
        self.frobenius_norm_dict = {}  # For logging Frobenius norms
        self._initial_frobenius_norms = {}  # For "initialize" mode: param_id -> R
        # QKV / TP
        self.split_qkv = split_qkv
        self.is_qkv_fn = is_qkv_fn
        self.qkv_split_shapes = qkv_split_shapes
        self.qkv_split_mode = qkv_split_mode
        # FC1 split for gated linear units
        self.split_fc1 = split_fc1
        self.is_fc1_fn = is_fc1_fn
        self.fc1_split_shapes = fc1_split_shapes
        # MoE expert split for GroupedMLP
        self.split_moe_experts = split_moe_experts
        self.is_grouped_moe_fn = is_grouped_moe_fn
        self.pg_collection = pg_collection
        self.tp_mode = tp_mode

        # Placeholder for scaled_orthogonalize_fn
        # MuonHyperball uses custom orthogonalize() method instead
        def scaled_orthogonalize_fn(grad: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError(
                "MuonHyperball uses custom orthogonalize() method. "
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
            log_per_module_update_rms=False,
        )

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Perform a single optimization step.

        Implements: W_{t+1} = R * Normalize(W_t - lr * R * Normalize(u_t))

        The base class step() applies W <- W - lr * Phi where Phi = R * Normalize(msign(M)).
        After that, this override projects each parameter back to its Frobenius sphere.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            The loss value if closure is provided, None otherwise.
        """
        self.frobenius_norm_dict.clear()
        loss = super().step(closure)

        # Project all parameters to Frobenius sphere: W <- R * W / ||W||_F
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                self._project_to_sphere(p)

        return loss

    def _get_target_radius(self, W: torch.Tensor, param_key: str) -> float:
        """Get target Frobenius radius for a weight tensor.

        For "initialize" mode, captures ||W_0||_F on first call and caches it.

        Args:
            W: Weight tensor.
            param_key: Unique key for caching (e.g., param_name or id).

        Returns:
            Target Frobenius radius R.
        """
        if self.radius_mode == "initialize":
            if param_key not in self._initial_frobenius_norms:
                R = torch.norm(W.float(), p="fro").item()
                self._initial_frobenius_norms[param_key] = R
            return self._initial_frobenius_norms[param_key]
        else:
            return _compute_frobenius_target_radius(
                shape=W.shape,
                radius_mode=self.radius_mode,
            )

    def _get_tp_info(self, p: torch.Tensor) -> Tuple[Any, Optional[int]]:
        """Extract TP group and partition dim from a parameter."""
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
        return tp_group, partition_dim

    def _retract_with_tp(
        self,
        W: torch.Tensor,
        target_radius: float,
        tp_group: Any,
        partition_dim: Optional[int],
    ) -> None:
        """Retract W in-place to Frobenius sphere, handling TP if needed."""
        from .spectral_ball_utils import _tp_world_and_rank, _tp_gather_along_dim, _tp_split_along_dim

        ws, _ = _tp_world_and_rank(tp_group)
        tp_enabled = tp_group is not None and partition_dim is not None and ws > 1

        if tp_enabled:
            W_full = _tp_gather_along_dim(W, tp_group, partition_dim)
            _apply_frobenius_retract(W_full, target_radius)
            W_local = _tp_split_along_dim(W_full, tp_group, partition_dim)
            W.copy_(W_local)
        else:
            _apply_frobenius_retract(W, target_radius)

    def _project_to_sphere(self, p: torch.Tensor) -> None:
        """Project parameter (or its components) to Frobenius sphere after update.

        Mirrors the split logic in orthogonalize() to project each component
        independently to its own Frobenius sphere.
        """
        tp_group, partition_dim = self._get_tp_info(p)
        param_name = getattr(p, 'param_name', None)

        # MoE expert splitting path
        if self.split_moe_experts and self.is_grouped_moe_fn is not None and self.is_grouped_moe_fn(p):
            num_local_experts = getattr(p, 'num_local_experts', None)
            if num_local_experts is not None and num_local_experts > 1:
                if 'weight1' in (param_name or ''):
                    is_gated = getattr(p, 'is_gated', False)
                    ffn_multiplier = 2 if is_gated else 1
                    out_dim, in_dim = p.shape
                    ffn_dim_per_expert = in_dim // (num_local_experts * ffn_multiplier)
                    W_reshaped = p.data.view(out_dim, num_local_experts, ffn_dim_per_expert * ffn_multiplier)
                    for expert_idx in range(num_local_experts):
                        W_expert = W_reshaped[:, expert_idx, :]
                        if self.split_fc1 and is_gated:
                            W_gate, W_up = torch.split(W_expert, [ffn_dim_per_expert, ffn_dim_per_expert], dim=1)
                            for suffix, Wc in [("gate", W_gate), ("up", W_up)]:
                                key = f"{param_name}.expert{expert_idx}.{suffix}"
                                R = self._get_target_radius(Wc.t(), key)
                                self._retract_with_tp(Wc.t(), R, tp_group, partition_dim)
                        else:
                            key = f"{param_name}.expert{expert_idx}"
                            R = self._get_target_radius(W_expert.t(), key)
                            self._retract_with_tp(W_expert.t(), R, tp_group, partition_dim)
                    return
                elif 'weight2' in (param_name or ''):
                    out_dim, in_dim = p.shape
                    ffn_dim_per_expert = out_dim // num_local_experts
                    W_reshaped = p.data.view(num_local_experts, ffn_dim_per_expert, in_dim)
                    for expert_idx in range(num_local_experts):
                        W_expert = W_reshaped[expert_idx, :, :]
                        key = f"{param_name}.expert{expert_idx}"
                        R = self._get_target_radius(W_expert.t(), key)
                        self._retract_with_tp(W_expert.t(), R, tp_group, partition_dim)
                    return

        # QKV splitting path
        if self.split_qkv and self.is_qkv_fn is not None and self.is_qkv_fn(p):
            out_dim, in_dim = p.shape
            split_sum = sum(self.qkv_split_shapes)
            num_groups = out_dim // split_sum
            component_names = ['q', 'k', 'v']
            q_dim_per_group, kv_channels, _ = self.qkv_split_shapes
            heads_per_group = q_dim_per_group // kv_channels
            W_view = p.data.view(num_groups, split_sum, in_dim)

            if self.qkv_split_mode == "group":
                for g in range(num_groups):
                    Wg_comps = torch.split(W_view[g], list(self.qkv_split_shapes), dim=0)
                    for idx, Wi in enumerate(Wg_comps):
                        key = f"{param_name}.g{g}.{component_names[idx]}"
                        R = self._get_target_radius(Wi, key)
                        self._retract_with_tp(Wi, R, tp_group, partition_dim)
            elif self.qkv_split_mode == "head":
                for g in range(num_groups):
                    Wg_comps = torch.split(W_view[g], list(self.qkv_split_shapes), dim=0)
                    W_q, W_k, W_v = Wg_comps
                    W_q_heads = W_q.view(heads_per_group, kv_channels, in_dim)
                    for h in range(heads_per_group):
                        key = f"{param_name}.g{g}.Q.h{h}"
                        R = self._get_target_radius(W_q_heads[h], key)
                        self._retract_with_tp(W_q_heads[h], R, tp_group, partition_dim)
                    for suffix, Wc in [("K", W_k), ("V", W_v)]:
                        key = f"{param_name}.g{g}.{suffix}"
                        R = self._get_target_radius(Wc, key)
                        self._retract_with_tp(Wc, R, tp_group, partition_dim)
            else:  # component mode
                W_q, W_k, W_v = torch.split(W_view, list(self.qkv_split_shapes), dim=1)
                for idx, Wi in enumerate([W_q, W_k, W_v]):
                    Wi_flat = Wi.reshape(-1, in_dim)
                    key = f"{param_name}.{component_names[idx]}"
                    R = self._get_target_radius(Wi_flat, key)
                    self._retract_with_tp(Wi_flat, R, tp_group, partition_dim)
            return

        # FC1 splitting path
        if self.split_fc1 and self.is_fc1_fn is not None and self.is_fc1_fn(p):
            gate_dim, up_dim = self.fc1_split_shapes
            W_gate, W_up = torch.split(p.data, [gate_dim, up_dim], dim=0)
            for suffix, Wc in [("gate", W_gate), ("up", W_up)]:
                key = f"{param_name}.{suffix}"
                R = self._get_target_radius(Wc, key)
                self._retract_with_tp(Wc, R, tp_group, partition_dim)
            return

        # Standard 2D matrix path
        key = param_name or str(id(p))
        R = self._get_target_radius(p.data, key)
        self._retract_with_tp(p.data, R, tp_group, partition_dim)

    def _compute_component_update(
        self,
        W: torch.Tensor,
        M: torch.Tensor,
        tp_group: Any,
        partition_dim: Optional[int],
        param_name: Optional[str] = None,
        component_label: Optional[str] = None,
    ) -> torch.Tensor:
        """Compute MuonHyperball update for a single component: R * Normalize(msign(M)).

        Args:
            W: Weight tensor for this component.
            M: Momentum tensor for this component.
            tp_group: Tensor parallel group.
            partition_dim: Partition dimension for TP.
            param_name: Parameter name for logging.
            component_label: Label like 'q', 'k', 'v' for logging.

        Returns:
            Update direction tensor: R * Normalize(msign(M)).
        """
        key = f"{param_name}.{component_label}" if param_name and component_label else str(id(W))
        R = self._get_target_radius(W, key)

        u, frob_norm = compute_muon_hyperball_update(
            W=W,
            M=M,
            target_radius=R,
            msign_steps=self.msign_steps,
            tp_group=tp_group,
            partition_dim=partition_dim,
        )

        # Record norm for logging
        if param_name and component_label:
            self.frobenius_norm_dict[f"{param_name}.{component_label}"] = frob_norm

        return u

    def orthogonalize(self, p: torch.Tensor, grad: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Compute MuonHyperball update direction.

        This method overrides the base class orthogonalize() to implement the
        MuonHyperball algorithm. The input 'grad' is actually the momentum M
        (potentially with Nesterov momentum applied by the base class).

        Args:
            p: Parameter tensor (current weight matrix W).
            grad: Momentum tensor M (after Nesterov if applicable).
            **kwargs: Additional parameters from param_group (includes 'lr').

        Returns:
            Update direction Φ to be applied as: W ← W - lr * Φ
        """
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

        param_name = getattr(p, 'param_name', None)

        # MoE expert splitting path for GroupedMLP
        if self.split_moe_experts and self.is_grouped_moe_fn is not None and self.is_grouped_moe_fn(p):
            num_local_experts = getattr(p, 'num_local_experts', None)
            if num_local_experts is not None and num_local_experts > 1:
                if 'weight1' in (param_name or ''):
                    is_gated = getattr(p, 'is_gated', False)
                    ffn_multiplier = 2 if is_gated else 1
                    out_dim, in_dim = p.shape
                    ffn_dim_per_expert = in_dim // (num_local_experts * ffn_multiplier)

                    W_reshaped = p.data.view(out_dim, num_local_experts, ffn_dim_per_expert * ffn_multiplier)
                    M_reshaped = grad.view(out_dim, num_local_experts, ffn_dim_per_expert * ffn_multiplier)

                    expert_updates = []
                    for expert_idx in range(num_local_experts):
                        W_expert = W_reshaped[:, expert_idx, :]
                        M_expert = M_reshaped[:, expert_idx, :]

                        if self.split_fc1 and is_gated:
                            W_gate, W_up = torch.split(W_expert, [ffn_dim_per_expert, ffn_dim_per_expert], dim=1)
                            M_gate, M_up = torch.split(M_expert, [ffn_dim_per_expert, ffn_dim_per_expert], dim=1)

                            U_gate = self._compute_component_update(
                                W_gate.t(), M_gate.t(), tp_group, partition_dim,
                                param_name, f'expert{expert_idx}.gate'
                            ).t()
                            U_up = self._compute_component_update(
                                W_up.t(), M_up.t(), tp_group, partition_dim,
                                param_name, f'expert{expert_idx}.up'
                            ).t()

                            U_expert = torch.cat([U_gate, U_up], dim=1)
                        else:
                            U_expert = self._compute_component_update(
                                W_expert.t(), M_expert.t(), tp_group, partition_dim,
                                param_name, f'expert{expert_idx}'
                            ).t()
                        expert_updates.append(U_expert)

                    update = torch.stack(expert_updates, dim=1).view(out_dim, in_dim)
                    return update

                elif 'weight2' in (param_name or ''):
                    out_dim, in_dim = p.shape
                    ffn_dim_per_expert = out_dim // num_local_experts

                    W_reshaped = p.data.view(num_local_experts, ffn_dim_per_expert, in_dim)
                    M_reshaped = grad.view(num_local_experts, ffn_dim_per_expert, in_dim)

                    expert_updates = []
                    for expert_idx in range(num_local_experts):
                        W_expert = W_reshaped[expert_idx, :, :]
                        M_expert = M_reshaped[expert_idx, :, :]

                        U_expert = self._compute_component_update(
                            W_expert.t(), M_expert.t(), tp_group, partition_dim,
                            param_name, f'expert{expert_idx}'
                        ).t()
                        expert_updates.append(U_expert)

                    update = torch.stack(expert_updates, dim=0).view(out_dim, in_dim)
                    return update

        # QKV splitting path
        if self.split_qkv and self.is_qkv_fn is not None and self.is_qkv_fn(p):
            assert self.qkv_split_shapes is not None, "qkv_split_shapes must be provided when split_qkv=True"
            out_dim, in_dim = p.shape
            split_sum = sum(self.qkv_split_shapes)
            assert (
                out_dim % split_sum == 0
            ), f"QKV split shapes {self.qkv_split_shapes} do not divide output dim {out_dim}"
            num_groups = out_dim // split_sum
            component_names = ['q', 'k', 'v']

            q_dim_per_group, kv_channels, _ = self.qkv_split_shapes
            heads_per_group = q_dim_per_group // kv_channels

            W_view = p.data.view(num_groups, split_sum, in_dim)
            M_view = grad.view(num_groups, split_sum, in_dim)

            if self.qkv_split_mode == "group":
                group_updates = []
                for g in range(num_groups):
                    Wg_comps = torch.split(W_view[g], list(self.qkv_split_shapes), dim=0)
                    Mg_comps = torch.split(M_view[g], list(self.qkv_split_shapes), dim=0)

                    comp_updates = []
                    for idx, (Wi, Mi) in enumerate(zip(Wg_comps, Mg_comps)):
                        label = f"g{g}.{component_names[idx]}"
                        ui = self._compute_component_update(Wi, Mi, tp_group, partition_dim, param_name, label)
                        comp_updates.append(ui)

                    group_updates.append(torch.cat(comp_updates, dim=0))

                update = torch.stack(group_updates, dim=0).reshape(out_dim, in_dim)
                return update

            elif self.qkv_split_mode == "head":
                group_updates = []
                for g in range(num_groups):
                    Wg_comps = torch.split(W_view[g], list(self.qkv_split_shapes), dim=0)
                    Mg_comps = torch.split(M_view[g], list(self.qkv_split_shapes), dim=0)

                    W_q, W_k, W_v = Wg_comps
                    M_q, M_k, M_v = Mg_comps

                    W_q_heads = W_q.view(heads_per_group, kv_channels, in_dim)
                    M_q_heads = M_q.view(heads_per_group, kv_channels, in_dim)

                    q_head_updates = []
                    for h in range(heads_per_group):
                        label = f"g{g}.Q.h{h}"
                        uh = self._compute_component_update(
                            W_q_heads[h], M_q_heads[h], tp_group, partition_dim,
                            param_name, label
                        )
                        q_head_updates.append(uh)

                    U_q = torch.stack(q_head_updates, dim=0).reshape(-1, in_dim)

                    U_k = self._compute_component_update(W_k, M_k, tp_group, partition_dim,
                                                        param_name, f"g{g}.K")
                    U_v = self._compute_component_update(W_v, M_v, tp_group, partition_dim,
                                                        param_name, f"g{g}.V")

                    group_updates.append(torch.cat([U_q, U_k, U_v], dim=0))

                update = torch.stack(group_updates, dim=0).reshape(out_dim, in_dim)
                return update

            else:  # component mode
                W_q, W_k, W_v = torch.split(W_view, list(self.qkv_split_shapes), dim=1)
                M_q, M_k, M_v = torch.split(M_view, list(self.qkv_split_shapes), dim=1)

                comps_W = [W_q.reshape(-1, in_dim), W_k.reshape(-1, in_dim), W_v.reshape(-1, in_dim)]
                comps_M = [M_q.reshape(-1, in_dim), M_k.reshape(-1, in_dim), M_v.reshape(-1, in_dim)]

                updates = []
                for idx, (Wi, Mi) in enumerate(zip(comps_W, comps_M)):
                    ui = self._compute_component_update(Wi, Mi, tp_group, partition_dim, param_name, component_names[idx])
                    part_out = self.qkv_split_shapes[idx]
                    updates.append(ui.view(num_groups, part_out, in_dim))

                U_q, U_k, U_v = updates
                update = torch.cat([U_q, U_k, U_v], dim=1).reshape(out_dim, in_dim)
                return update

        # FC1 splitting path for gated linear units (SwiGLU)
        if self.split_fc1 and self.is_fc1_fn is not None and self.is_fc1_fn(p):
            assert self.fc1_split_shapes is not None, "fc1_split_shapes must be provided when split_fc1=True"
            out_dim, in_dim = p.shape
            gate_dim, up_dim = self.fc1_split_shapes
            assert (
                out_dim == gate_dim + up_dim
            ), f"FC1 split shapes {self.fc1_split_shapes} do not match output dim {out_dim}"

            W_gate, W_up = torch.split(p.data, [gate_dim, up_dim], dim=0)
            M_gate, M_up = torch.split(grad, [gate_dim, up_dim], dim=0)

            U_gate = self._compute_component_update(W_gate, M_gate, tp_group, partition_dim, param_name, "gate")
            U_up = self._compute_component_update(W_up, M_up, tp_group, partition_dim, param_name, "up")

            update = torch.cat([U_gate, U_up], dim=0)
            return update

        # Standard 2D matrix path
        key = param_name or str(id(p))
        target_radius = self._get_target_radius(p.data, key)

        update, frob_norm = compute_muon_hyperball_update(
            W=p.data,
            M=grad,
            target_radius=target_radius,
            msign_steps=self.msign_steps,
            tp_group=tp_group,
            partition_dim=partition_dim,
        )

        # Record norm for logging
        if param_name:
            self.frobenius_norm_dict[param_name] = frob_norm

        return update

    def get_frobenius_norm_dict(self):
        """Get Frobenius norm dictionary for logging.

        Returns:
            Dictionary mapping module names to their Frobenius norm values,
            or None if dict is empty.
        """
        if not self.frobenius_norm_dict:
            return None
        return self.frobenius_norm_dict


MuonHyperball.__doc__ = MuonHyperball.__doc__.format(_args_doc=_args_doc)  # type: ignore[union-attr]
