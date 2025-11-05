# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Base utilities for spectral ball optimizer solvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from .msign import msign


# -----------------------------------------------------------------------------
# Basic utilities (FP32 accumulation to reduce mixed-precision error)
# -----------------------------------------------------------------------------
@torch.no_grad()
def inner_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Frobenius inner product <a, b>, returned as a scalar tensor on GPU."""
    return (a.to(torch.float32) * b.to(torch.float32)).sum()


@torch.no_grad()
def trace_fp32(a: torch.Tensor) -> torch.Tensor:
    """Trace with FP32 accumulation; returns a scalar tensor on GPU."""
    return torch.trace(a.to(torch.float32))


# -----------------------------------------------------------------------------
# Objective helpers (decoupled, solver-agnostic)
#   - compute_phi(G, Θ, λ): Φ(λ) = msign(G + λΘ)
#   - compute_f(G, Θ, λ): f(λ) = <Θ, Φ(λ)>
#   Legacy convenience evaluate_objective_and_stats kept for compatibility.
# -----------------------------------------------------------------------------
@torch.no_grad()
def compute_phi(
    G: torch.Tensor, Theta: torch.Tensor, lambda_value: float, msign_steps: int = 5
) -> torch.Tensor:
    """Compute Φ(λ) = msign(G + λΘ) on GPU (no SVD)."""
    device = G.device
    lambda_tensor = torch.tensor(lambda_value, device=device, dtype=torch.float32)
    Z = G + lambda_tensor * Theta
    return msign(Z, steps=msign_steps)


@torch.no_grad()
def compute_f(
    G: torch.Tensor, Theta: torch.Tensor, lambda_value: float, msign_steps: int = 5
) -> float:
    """Compute scalar f(λ) = <Θ, Φ(λ)> with Φ(λ)=msign(G+λΘ)."""
    Phi = compute_phi(G, Theta, lambda_value, msign_steps)
    return float(inner_product(Theta, Phi).item())


@torch.no_grad()
def evaluate_objective_and_stats(
    G: torch.Tensor, Theta: torch.Tensor, lambda_value: float, msign_steps: int = 5
) -> Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Legacy helper retained for compatibility.
    Returns (f, Φ, X=ΘᵀΦ, q=ZᵀΦ) where Z=G+λΘ.
    """
    device = G.device
    lambda_tensor = torch.tensor(lambda_value, device=device, dtype=torch.float32)
    Z = G + lambda_tensor * Theta
    Phi = msign(Z, steps=msign_steps)
    X = Theta.mT @ Phi
    q = Z.mT @ Phi
    f_val = float(inner_product(Theta, Phi).item())
    return f_val, Phi, X, q


# -----------------------------------------------------------------------------
# Standard result record
# -----------------------------------------------------------------------------
@dataclass
class SolverResult:
    """Unified solver result structure.

    - method: Solver name ('brent'/'bisection'/'secant'/'fixed_point'/'newton').
    - solution: Final λ.
    - residual: Final |f(λ)|.
    - iterations: Iteration count (the only counting metric we care about).
    - converged: Whether convergence criterion was met (defined in each solver).
    - time_sec: Solve time (seconds).
    - bracket: Optional, bracket interval (lo, hi).
    - history: Optional, contains 'solution' and 'residual' trajectories.
    """

    method: str
    solution: float
    residual: float
    iterations: int
    converged: bool
    time_sec: float = 0.0
    bracket: tuple[float, float] | None = None
