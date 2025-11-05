# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Brent's method for solving the Lagrange multiplier in spectral ball optimization."""

from __future__ import annotations

import time
from typing import Tuple

import torch

from .base import SolverResult, compute_f


"""
Brent's method: Solve f(λ)=⟨Θ, Φ(λ)⟩=0, where Φ(λ)=msign(G+λΘ).

Prerequisites and workflow:
  - Requires a bracket interval [a,b] such that f(a)·f(b)≤0 (see find_bracket).
  - Iteratively combines bisection, secant, and inverse quadratic interpolation for robustness and efficiency.

Convergence criterion (function value only):
  - Only when |f(λ)| ≤ tolerance_f is convergence declared; otherwise, reaching iteration limit is considered non-convergent.

Statistics: Only counts iteration steps; function evaluation count is no longer tracked.
"""


# -----------------------------------------------------------------------------
# Bracketing: find [a, b] such that f(a) * f(b) <= 0
# -----------------------------------------------------------------------------
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
        G: Momentum tensor (not raw gradient!).
        Theta: Outer product of top singular vectors (u1 @ v1^T).
        initial_guess: Starting point for bracketing search.
        initial_step: Initial step size for expansion.
        max_expansions: Maximum number of expansion attempts.
        msign_steps: Number of Newton-Schulz iterations for msign.

    Returns:
        Tuple (a, b, fa, fb) where a <= b and f(a) * f(b) <= 0 if successful.
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


# -----------------------------------------------------------------------------
# Brent root-finding; function values are evaluated on GPU
# -----------------------------------------------------------------------------
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
        G: Momentum tensor (not raw gradient!).
        Theta: Outer product of top singular vectors (u1 @ v1^T).
        a: Left bracket endpoint.
        b: Right bracket endpoint.
        fa: f(a).
        fb: f(b).
        tolerance_f: Function value tolerance for convergence.
        tolerance_x: Variable tolerance for convergence.
        max_iterations: Maximum iteration count.
        msign_steps: Number of Newton-Schulz iterations for msign.

    Returns:
        SolverResult containing solution, residual, and convergence info.
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


# -----------------------------------------------------------------------------
# High-level helper: bracket + Brent solver wrapped together
# -----------------------------------------------------------------------------
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

    `G` is expected to be the first momentum `M`, not the raw gradient. `Theta` should be the
    outer product of the leading singular vectors (`u1 v1^T`). All tensors stay on the current
    device to avoid host/device thrash.

    Args:
        G: Momentum tensor (not raw gradient!).
        Theta: Outer product of top singular vectors (u1 @ v1^T).
        initial_guess: Starting point for bracketing.
        initial_step: Initial step size for bracketing.
        tolerance_f: Function value tolerance.
        tolerance_x: Variable tolerance.
        max_iterations: Maximum Brent iterations.
        max_expansions: Maximum bracketing expansions.
        msign_steps: Number of Newton-Schulz iterations for msign.

    Returns:
        SolverResult containing solution, residual, and convergence info.
    """
    a, b, fa, fb = find_bracket(
        G,
        Theta,
        initial_guess=initial_guess,
        initial_step=initial_step,
        max_expansions=max_expansions,
        msign_steps=msign_steps,
    )

    # If the bracket degenerates (no sign change found), report failure with the best residual.
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
