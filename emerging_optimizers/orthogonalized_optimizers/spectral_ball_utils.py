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

"""Utility functions for Spectral Ball optimizer."""

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from absl import logging

from .muon_utils import newton_schulz


__all__ = ["compute_target_radius", "compute_spectral_ball_update"]


# =============================================================================
# Power Iteration
# =============================================================================
@torch.no_grad()
def power_iteration(w: torch.Tensor, steps: int = 10, eps: float = 1e-20):
    """Return leading singular triplet (σ, u, v) via bilateral power iteration.

    All tensors are float32 on the same device as w.

    Args:
        w: Weight matrix tensor
        steps: Number of power iteration steps
        eps: Small epsilon to avoid division by zero

    Returns:
        Tuple of (sigma, u, v) where sigma is the top singular value,
        u is the left singular vector, v is the right singular vector
    """
    w = w.to(torch.float32)
    gram = w.transpose(-2, -1).matmul(w)  # precompute W^T W
    v = torch.ones(*w.shape[:-2], w.shape[-1], 1, device=w.device, dtype=w.dtype)
    for _ in range(steps):
        v = gram.matmul(v)
        v = v / torch.clamp(torch.linalg.vector_norm(v, ord=2, dim=-2, keepdim=True), min=eps)
    u = w.matmul(v)
    u = u / torch.clamp(torch.linalg.vector_norm(u, ord=2, dim=-2, keepdim=True), min=eps)
    s = (u.transpose(-2, -1).matmul(w).matmul(v)).squeeze(-1).squeeze(-1)
    return s, u, v


# =============================================================================
# Brent Solver for Lagrange Multiplier
# =============================================================================
@dataclass
class SolverResult:
    """Unified solver result structure.

    Attributes:
        method: Solver name (e.g., 'brent')
        solution: Final λ value
        residual: Final |f(λ)|
        iterations: Number of iterations
        converged: Whether convergence criterion was met
        time_sec: Solve time in seconds
        bracket: Optional bracket interval (lo, hi)
    """
    method: str
    solution: float
    residual: float
    iterations: int
    converged: bool
    time_sec: float = 0.0
    bracket: Optional[Tuple[float, float]] = None


@torch.no_grad()
def inner_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Frobenius inner product <a, b>, returned as a scalar tensor on GPU."""
    return (a.to(torch.float32) * b.to(torch.float32)).sum()


@torch.no_grad()
def compute_phi(
    G: torch.Tensor,
    Theta: torch.Tensor,
    lambda_value: float,
    msign_steps: int = 5
) -> torch.Tensor:
    """Compute Φ(λ) = msign(G + λΘ) using Newton-Schulz iteration.

    Args:
        G: Momentum tensor
        Theta: Outer product of top singular vectors
        lambda_value: Lagrange multiplier value
        msign_steps: Number of Newton-Schulz iterations

    Returns:
        Φ(λ) computed via Newton-Schulz iteration
    """
    device = G.device
    lambda_tensor = torch.tensor(lambda_value, device=device, dtype=torch.float32)
    Z = G + lambda_tensor * Theta

    # Use newton_schulz with polar_express coefficients
    # newton_schulz expects float32 input
    Z_fp32 = Z.to(torch.float32)
    Phi = newton_schulz(
        Z_fp32,
        steps=msign_steps,
        coefficient_type="polar_express",
        eps=1e-7,
        transpose=None,  # Let it auto-determine
        tp_group=None,
        use_syrk=False,
    )
    return Phi


@torch.no_grad()
def compute_f(
    G: torch.Tensor,
    Theta: torch.Tensor,
    lambda_value: float,
    msign_steps: int = 5
) -> float:
    """Compute scalar f(λ) = <Θ, Φ(λ)> with Φ(λ)=msign(G+λΘ).

    Args:
        G: Momentum tensor
        Theta: Outer product of top singular vectors
        lambda_value: Lagrange multiplier value
        msign_steps: Number of Newton-Schulz iterations

    Returns:
        Scalar value f(λ)
    """
    Phi = compute_phi(G, Theta, lambda_value, msign_steps)
    return float(inner_product(Theta, Phi).item())


@torch.no_grad()
def find_bracket(
    G: torch.Tensor,
    Theta: torch.Tensor,
    initial_guess: float = 0.0,
    initial_step: float = 1.0,
    max_expansions: int = 60,
    msign_steps: int = 5,
) -> Tuple[float, float, float, float]:
    """Find a bracket interval [a, b] where f(a) * f(b) <= 0.

    Args:
        G: Momentum tensor
        Theta: Outer product of top singular vectors
        initial_guess: Starting point for bracketing search
        initial_step: Initial step size for expansion
        max_expansions: Maximum number of expansion attempts
        msign_steps: Number of Newton-Schulz iterations

    Returns:
        Tuple (a, b, fa, fb) where a <= b and f(a) * f(b) <= 0 if successful
    """
    fa = compute_f(G, Theta, initial_guess, msign_steps)
    if fa == 0.0:
        return initial_guess, initial_guess, fa, fa

    step = initial_step if initial_step > 0 else 1.0
    a = b = initial_guess
    fb = fa

    for _ in range(max_expansions):
        # Try right
        b = initial_guess + step
        fb = compute_f(G, Theta, b, msign_steps)
        if fa * fb <= 0:
            return (a, b, fa, fb) if a <= b else (b, a, fb, fa)

        # Try left
        a = initial_guess - step
        fa = compute_f(G, Theta, a, msign_steps)
        if fa * fb <= 0:
            return (a, b, fa, fb) if a <= b else (b, a, fb, fa)

        step *= 2.0

    return min(a, b), max(a, b), fa, fb


@torch.no_grad()
def solve_with_brent(
    G: torch.Tensor,
    Theta: torch.Tensor,
    a: float,
    b: float,
    fa: float,
    fb: float,
    tolerance_f: float = 1e-8,
    tolerance_x: float = 1e-10,
    max_iterations: int = 100,
    msign_steps: int = 5,
) -> SolverResult:
    """Solve for λ using Brent's method given a bracket [a, b].

    Args:
        G: Momentum tensor
        Theta: Outer product of top singular vectors
        a: Left bracket endpoint
        b: Right bracket endpoint
        fa: f(a)
        fb: f(b)
        tolerance_f: Function value tolerance for convergence
        tolerance_x: Variable tolerance for convergence
        max_iterations: Maximum iteration count
        msign_steps: Number of Newton-Schulz iterations

    Returns:
        SolverResult containing solution, residual, and convergence info
    """
    start = time.perf_counter()

    if fa == 0.0:
        return SolverResult("brent", a, 0.0, 0, True, 0.0, (a, b))
    if fb == 0.0:
        return SolverResult("brent", b, 0.0, 0, True, 0.0, (a, b))

    c, fc = a, fa
    d = e = b - a
    x, fx = b, fb

    for it in range(1, max_iterations + 1):
        if fx == 0.0:
            return SolverResult(
                "brent", x, 0.0, it, True, time.perf_counter() - start, (a, b)
            )
        if fa * fb > 0:
            a, fa = c, fc
            d = e = b - a
        if abs(fa) < abs(fb):
            a, fa, b, fb, c, fc = c, fc, a, fa, b, fb

        tol = 2.0 * tolerance_x * max(1.0, abs(b))
        m = 0.5 * (c - b)

        if abs(fb) <= tolerance_f:
            return SolverResult(
                "brent", b, abs(fb), it, True, time.perf_counter() - start, (a, b)
            )

        if abs(e) >= tol and abs(fc) > abs(fb):
            s = fb / fc
            if a == c:
                p = 2.0 * m * s
                q = 1.0 - s
            else:
                q = fc / fa
                r = fb / fa
                p = s * (2.0 * m * q * (q - r) - (b - c) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0:
                q = -q
            p = abs(p)
            if 2.0 * p < min(3.0 * m * q - abs(tol * q), abs(e * q)):
                e, d = d, p / q
            else:
                d = e = m
        else:
            d = e = m

        c, fc = b, fb
        if abs(d) > tol:
            b += d
        else:
            b += tol if m > 0 else -tol

        fb = compute_f(G, Theta, b, msign_steps)

    return SolverResult(
        "brent", b, abs(fb), max_iterations, False, time.perf_counter() - start, (a, b)
    )


@torch.no_grad()
def solve_lambda_with_brent(
    G: torch.Tensor,
    Theta: torch.Tensor,
    initial_guess: float = 0.0,
    initial_step: float = 1.0,
    tolerance_f: float = 1e-8,
    tolerance_x: float = 1e-10,
    max_iterations: int = 100,
    max_expansions: int = 60,
    msign_steps: int = 5,
) -> SolverResult:
    """Full λ solver: find a bracket then run Brent iterations.

    Args:
        G: Momentum tensor (first momentum M, not raw gradient)
        Theta: Outer product of top singular vectors (u @ v^T)
        initial_guess: Starting point for bracketing
        initial_step: Initial step size for bracketing
        tolerance_f: Function value tolerance
        tolerance_x: Variable tolerance
        max_iterations: Maximum Brent iterations
        max_expansions: Maximum bracketing expansions
        msign_steps: Number of Newton-Schulz iterations

    Returns:
        SolverResult containing solution, residual, and convergence info
    """
    a, b, fa, fb = find_bracket(
        G,
        Theta,
        initial_guess=initial_guess,
        initial_step=initial_step,
        max_expansions=max_expansions,
        msign_steps=msign_steps,
    )

    # If the bracket degenerates (no sign change found), report failure
    if fa * fb > 0:
        residual = min(abs(fa), abs(fb))
        return SolverResult("brent", initial_guess, residual, 0, False, 0.0, (a, b))

    return solve_with_brent(
        G,
        Theta,
        a=a,
        b=b,
        fa=fa,
        fb=fb,
        tolerance_f=tolerance_f,
        tolerance_x=tolerance_x,
        max_iterations=max_iterations,
        msign_steps=msign_steps,
    )


# =============================================================================
# Target Radius Computation
# =============================================================================
def compute_target_radius(
    shape: tuple,
    radius_mode: str,
    current_weight: Optional[torch.Tensor] = None,
) -> float:
    """Compute target spectral norm radius R based on radius_mode.

    The radius determines the size of the spectral sphere constraint ||W||_2 = R.

    Args:
        shape: Parameter shape tuple (n_out, n_in)
        radius_mode: One of:
            - "spectral_mup": R = sqrt(n_out / n_in)
                Used for μP-style scaling where update magnitudes scale properly
            - "identity": R = 1.0
                Standard unit spectral norm constraint
            - "initialize": R = ||W||_2 at initialization
                Preserves the initial scale of the weight matrix
        current_weight: Current weight tensor, required only for "initialize" mode

    Returns:
        Target radius R as a float

    Raises:
        ValueError: If radius_mode is invalid or current_weight is None for "initialize" mode
    """
    if radius_mode == "spectral_mup":
        n_out, n_in = shape
        return math.sqrt(n_out / n_in)
    elif radius_mode == "identity":
        return 1.0
    elif radius_mode == "initialize":
        if current_weight is None:
            raise ValueError("current_weight is required for radius_mode='initialize'")
        # Use power iteration to compute initial spectral norm
        # Use more steps for accurate initialization
        sigma, _, _ = power_iteration(current_weight, steps=20)
        radius = float(sigma.item())
        if radius <= 0:
            logging.warning(f"Computed radius {radius} <= 0, falling back to 1.0")
            radius = 1.0
        return radius
    else:
        raise ValueError(
            f"Invalid radius_mode: {radius_mode}. "
            f"Must be one of: spectral_mup, identity, initialize"
        )


# =============================================================================
# Core Spectral Ball Update
# =============================================================================
def compute_spectral_ball_update(
    W: torch.Tensor,
    M: torch.Tensor,
    target_radius: float,
    power_iteration_steps: int,
    msign_steps: int,
    msign_coefficient_type: str,
    brent_tolerance_f: float,
    brent_tolerance_x: float,
    brent_max_iterations: int,
) -> torch.Tensor:
    """Compute spectral ball constrained update direction.

    This function implements the core spectral ball optimization algorithm:

    1. **Unified Power Iteration for Retraction and Theta**:
       - Compute σ, u, v = power_iteration(W, steps)
       - This gives us the spectral norm σ and top singular vectors (u, v)

    2. **Retraction to Spectral Sphere** (in-place):
       - W ← (R/σ) * W
       - Projects W onto the spectral sphere of radius R
       - This is the "A" operation in the A-B-A-B sequence

    3. **Form Theta**:
       - Θ = u @ v^T
       - This is the rank-1 matrix formed by the top singular vectors

    4. **Solve for Lagrange Multiplier λ**:
       - Find λ such that <Θ, msign(M + λΘ)> = 0
       - Uses Brent's method for robust root finding

    5. **Compute Update Direction**:
       - Φ = msign(M + λΘ)
       - This is the constrained gradient direction

    The key insight: The retraction at the end of step t equals the retraction at
    the beginning of step t+1, allowing us to fuse these operations.

    Args:
        W: Current weight matrix (will be modified in-place for retraction)
        M: Momentum tensor (after Nesterov momentum if applicable)
        target_radius: Target spectral norm R
        power_iteration_steps: Number of power iteration steps
        msign_steps: Number of Newton-Schulz iterations
        msign_coefficient_type: Coefficient type for msign (unused, uses polar_express)
        brent_tolerance_f: Function tolerance for Brent solver
        brent_tolerance_x: Variable tolerance for Brent solver
        brent_max_iterations: Maximum Brent iterations

    Returns:
        Update direction Φ to be applied as W ← W - lr * Φ

    Note:
        The parameter W is modified in-place during the retraction step.
        This is intentional and corresponds to the mathematical formulation.
    """
    # Step 1: Unified power iteration for both retraction and Theta
    # This computes the spectral norm and top singular vectors
    sigma, u, v = power_iteration(W, steps=power_iteration_steps)

    # Step 2: Retract W to spectral sphere (in-place modification)
    # This ensures ||W||_2 = target_radius
    sigma_value = sigma.item()
    if sigma_value > 0:
        scale_factor = target_radius / sigma_value
        W.mul_(scale_factor)
        logging.debug(
            f"Retracted W: sigma={sigma_value:.6f}, target={target_radius:.6f}, "
            f"scale={scale_factor:.6f}"
        )
    else:
        logging.warning(f"Singular value sigma={sigma_value} <= 0, skipping retraction")

    # Step 3: Form Theta = u @ v^T
    # This is the rank-1 matrix corresponding to the top singular direction
    Theta = u @ v.transpose(-2, -1)

    # Step 4: Solve for lambda using Brent's method
    # Find λ such that f(λ) = <Θ, msign(M + λΘ)> = 0
    result = solve_lambda_with_brent(
        G=M,
        Theta=Theta,
        initial_guess=0.0,  # Always start from 0 (no warm-start per user's instruction)
        initial_step=1.0,
        tolerance_f=brent_tolerance_f,
        tolerance_x=brent_tolerance_x,
        max_iterations=brent_max_iterations,
        max_expansions=60,
        msign_steps=msign_steps,
    )

    lambda_value = result.solution

    logging.debug(
        f"Brent solver: λ={lambda_value:.6f}, residual={result.residual:.2e}, "
        f"iterations={result.iterations}, converged={result.converged}"
    )

    if not result.converged:
        logging.warning(
            f"Brent solver did not converge: residual={result.residual:.2e} "
            f"after {result.iterations} iterations"
        )

    # Step 5: Compute update direction Φ = msign(M + λΘ)
    Z = M + lambda_value * Theta
    Phi = newton_schulz(
        Z.to(torch.float32),
        steps=msign_steps,
        coefficient_type="polar_express",  # Use polar_express as default
        eps=1e-7,
        transpose=None,
        tp_group=None,
        use_syrk=False,
    )

    return Phi
