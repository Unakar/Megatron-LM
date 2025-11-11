"""Utility functions for Spectral Ball optimizer."""

import math
import time
from dataclasses import dataclass
from itertools import chain, islice, repeat
from typing import Optional, Tuple

import torch
from absl import logging


__all__ = ["compute_target_radius", "compute_spectral_ball_update"]


# =============================================================================
# Matrix Sign Function (msign) Implementation
# =============================================================================

# Polar-Express coefficients for Newton-Schulz iteration
_POLAR_EXPRESS_COEFFS = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
]


def _deflate_coeffs(abc: tuple, deflation_eps: float) -> tuple:
    """Deflate coefficients for numerical stability."""
    a, b, c = abc
    return (
        a / (1 + deflation_eps),
        b / (1 + deflation_eps) ** 3,
        c / (1 + deflation_eps) ** 5,
    )


@torch.compile
def msign(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Compute matrix sign function via Newton-Schulz iteration with Polar-Express coefficients.

    This is the core matrix sign computation for the spectral ball optimizer.
    Uses deflated Polar-Express coefficients for fast convergence.

    Args:
        G: Input matrix (fp32 or bf16)
        steps: Number of Newton-Schulz iterations

    Returns:
        Matrix sign of G (same dtype as input)
    """
    assert G.ndim >= 2, "Input tensor must have at least two dimensions."
    assert steps > 0, "Number of steps must be positive."

    deflation_eps = 0.01
    X = G.bfloat16()

    # Handle tall matrices: transpose to make wide
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Normalize spectral norm to at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1 + deflation_eps) + 1e-7)

    # Precompute deflated coefficients (CPU operation outside loop)
    hs = [
        _deflate_coeffs(coeffs, deflation_eps)
        for coeffs in chain(
            islice(_POLAR_EXPRESS_COEFFS, steps),
            repeat(_POLAR_EXPRESS_COEFFS[-1], max(0, steps - len(_POLAR_EXPRESS_COEFFS))),
        )
    ]

    # Newton-Schulz iteration (GPU operations only)
    for a, b, c in hs:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    # Transpose back if needed
    if G.size(-2) > G.size(-1):
        X = X.mT

    return X


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
    """Frobenius inner product <a, b>, returned as a scalar tensor on GPU.

    Args:
        a, b: Input tensors (assumed to be fp32 for numerical stability)

    Returns:
        Scalar tensor containing <a, b>
    """
    return (a * b).sum()


@torch.no_grad()
def compute_phi(
    G: torch.Tensor,
    Theta: torch.Tensor,
    lambda_value: float,
    msign_steps: int = 5
) -> torch.Tensor:
    """Compute Φ(λ) = msign(G + λΘ) using Newton-Schulz iteration.

    Args:
        G: Momentum tensor (fp32)
        Theta: Outer product of top singular vectors (fp32)
        lambda_value: Lagrange multiplier value
        msign_steps: Number of Newton-Schulz iterations

    Returns:
        Φ(λ) computed via Newton-Schulz iteration (fp32)

    Note:
        Assumes G and Theta are already in fp32 to avoid redundant conversions.
    """
    Z = G + lambda_value * Theta
    Phi = msign(Z, steps=msign_steps)
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
        G: Momentum tensor (fp32)
        Theta: Outer product of top singular vectors (fp32)
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
@torch.no_grad()
def _tp_world_and_rank(tp_group: torch.distributed.ProcessGroup | None) -> tuple[int, int]:
    if tp_group is None:
        return 1, 0
    return tp_group.size(), tp_group.rank()


@torch.no_grad()
def _tp_gather_along_dim(
    x: torch.Tensor,
    tp_group: torch.distributed.ProcessGroup,
    dim: int,
) -> torch.Tensor:
    """All-gather shards along `dim` and concatenate into a global tensor."""
    ws, _ = _tp_world_and_rank(tp_group)
    if ws == 1:
        return x
    shards = [torch.empty_like(x) for _ in range(ws)]
    torch.distributed.all_gather(shards, x, group=tp_group)
    return torch.cat(shards, dim=dim)


@torch.no_grad()
def _tp_split_along_dim(
    x_full: torch.Tensor,
    tp_group: torch.distributed.ProcessGroup,
    dim: int,
) -> torch.Tensor:
    """Split global tensor along `dim` and return the local shard for this rank."""
    ws, rk = _tp_world_and_rank(tp_group)
    if ws == 1:
        return x_full
    parts = x_full.chunk(ws, dim=dim)
    return parts[rk].contiguous()


def _compute_single_rank(
    W: torch.Tensor,
    M: torch.Tensor,
    target_radius: float,
    power_iteration_steps: int,
    msign_steps: int,
    brent_tolerance_f: float,
    brent_tolerance_x: float,
    brent_max_iterations: int,
) -> torch.Tensor:
    """Compute spectral ball update for single-rank (non-TP) case.

    This implements the core algorithm:
    1. Power iteration to get σ, u, v
    2. Retract W to spectral sphere: W ← (R/σ)W
    3. Form Θ = uv^T
    4. Solve for λ: <Θ, msign(M + λΘ)> = 0
    5. Return Φ = msign(M + λΘ)

    Args:
        W: Weight matrix (modified in-place for retraction)
        M: Momentum tensor
        target_radius: Target spectral norm R
        power_iteration_steps: Number of power iteration steps
        msign_steps: Number of Newton-Schulz iterations
        brent_tolerance_f: Function tolerance for Brent solver
        brent_tolerance_x: Variable tolerance for Brent solver
        brent_max_iterations: Maximum Brent iterations

    Returns:
        Update direction Φ (fp32)
    """
    # Convert M to fp32 once at the beginning
    M_fp32 = M.to(torch.float32)

    # 1. Power iteration (returns fp32)
    sigma, u, v = power_iteration(W, steps=power_iteration_steps)
    sigma_value = sigma.item()

    # 2. Retract W to spectral sphere
    if sigma_value > 0:
        scale_factor = target_radius / sigma_value
        W.mul_(scale_factor)
        logging.debug(
            f"Retracted W: sigma={sigma_value:.6f}, target={target_radius:.6f}, "
            f"scale={scale_factor:.6f}"
        )
    else:
        logging.warning(f"Singular value sigma={sigma_value} <= 0, skipping retraction")

    # 3. Form Theta (fp32)
    Theta = u @ v.transpose(-2, -1)

    # 4. Solve for lambda
    result = solve_lambda_with_brent(
        G=M_fp32,
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
    if not result.converged:
        logging.warning(
            f"Brent solver did not converge: residual={result.residual:.2e} "
            f"after {result.iterations} iterations"
        )

    # 5. Compute final update direction
    Z = M_fp32 + lambda_value * Theta
    Phi = msign(Z, steps=msign_steps)
    return Phi


def _compute_tp_duplicated(
    W: torch.Tensor,
    M: torch.Tensor,
    target_radius: float,
    power_iteration_steps: int,
    msign_steps: int,
    brent_tolerance_f: float,
    brent_tolerance_x: float,
    brent_max_iterations: int,
    tp_group: torch.distributed.ProcessGroup,
    partition_dim: int,
) -> torch.Tensor:
    """Compute spectral ball update for TP duplicated mode.

    Communication pattern (optimal):
    1. all_gather(W_shard) → W_full
    2. all_gather(M_shard) → M_full
    3. Compute on full tensors (no communication)
    4. Split Φ_full → Φ_local (local operation)

    Total: 2 all_gather operations

    Args:
        W: Weight matrix shard (modified in-place for retraction)
        M: Momentum tensor shard
        target_radius: Target spectral norm R
        power_iteration_steps: Number of power iteration steps
        msign_steps: Number of Newton-Schulz iterations
        brent_tolerance_f: Function tolerance for Brent solver
        brent_tolerance_x: Variable tolerance for Brent solver
        brent_max_iterations: Maximum Brent iterations
        tp_group: Tensor parallel process group
        partition_dim: Dimension along which tensors are partitioned

    Returns:
        Update direction Φ_local (fp32 shard)
    """
    # Gather shards to global matrices
    W_full = _tp_gather_along_dim(W, tp_group, partition_dim)
    M_full = _tp_gather_along_dim(M, tp_group, partition_dim)

    # Convert M to fp32 once
    M_full_fp32 = M_full.to(torch.float32)

    # 1. Power iteration on global W (returns fp32)
    sigma, u, v = power_iteration(W_full, steps=power_iteration_steps)
    sigma_value = sigma.item()

    # 2. Retract global W and update local shard
    if sigma_value > 0:
        scale_factor = target_radius / sigma_value
        W_full_retracted = W_full * scale_factor
        # Split back to local shard and update original W
        W_local = _tp_split_along_dim(W_full_retracted, tp_group, partition_dim)
        W.copy_(W_local)
        logging.debug(
            f"[TP] Retracted W: sigma={sigma_value:.6f}, target={target_radius:.6f}, "
            f"scale={scale_factor:.6f}"
        )
    else:
        logging.warning(
            f"[TP] Singular value sigma={sigma_value} <= 0, skipping retraction"
        )

    # 3. Form Theta (fp32)
    Theta_full = u @ v.transpose(-2, -1)

    # 4. Solve for lambda on global tensors
    result = solve_lambda_with_brent(
        G=M_full_fp32,
        Theta=Theta_full,
        initial_guess=0.0,
        initial_step=1.0,
        tolerance_f=brent_tolerance_f,
        tolerance_x=brent_tolerance_x,
        max_iterations=brent_max_iterations,
        max_expansions=60,
        msign_steps=msign_steps,
    )
    lambda_value = result.solution
    if not result.converged:
        logging.warning(
            f"[TP] Brent solver did not converge: residual={result.residual:.2e} "
            f"after {result.iterations} iterations"
        )

    # 5. Compute Φ on global tensor (no communication)
    Z_full = M_full_fp32 + lambda_value * Theta_full
    Phi_full = msign(Z_full, steps=msign_steps)

    # 6. Split back to local shard
    Phi_local = _tp_split_along_dim(Phi_full, tp_group, partition_dim)
    return Phi_local


def compute_spectral_ball_update(
    W: torch.Tensor,
    M: torch.Tensor,
    target_radius: float,
    power_iteration_steps: int,
    msign_steps: int,
    brent_tolerance_f: float,
    brent_tolerance_x: float,
    brent_max_iterations: int,
    *,
    tp_group: torch.distributed.ProcessGroup | None = None,
    partition_dim: int | None = None,
    tp_mode: str = "duplicated",
) -> torch.Tensor:
    """Compute spectral ball constrained update direction (dispatcher).

    This is the main entry point that dispatches to either single-rank or
    tensor-parallel implementations based on the TP configuration.

    Algorithm overview:
    1. Power iteration to get σ, u, v
    2. Retract W to spectral sphere: W ← (R/σ)W
    3. Form Θ = uv^T
    4. Solve for λ: <Θ, msign(M + λΘ)> = 0
    5. Return Φ = msign(M + λΘ)

    The msign function uses Polar-Express coefficients for fast convergence.
    See _compute_single_rank and _compute_tp_duplicated for implementation details.

    Args:
        W: Current weight matrix (modified in-place for retraction)
        M: Momentum tensor
        target_radius: Target spectral norm R
        power_iteration_steps: Number of power iteration steps
        msign_steps: Number of Newton-Schulz iterations (uses Polar-Express coefficients)
        brent_tolerance_f: Function tolerance for Brent solver
        brent_tolerance_x: Variable tolerance for Brent solver
        brent_max_iterations: Maximum Brent iterations
        tp_group: Tensor parallel process group (None for single-rank)
        partition_dim: Dimension along which tensors are partitioned
        tp_mode: TP mode (only "duplicated" is currently supported)

    Returns:
        Update direction Φ to be applied as W ← W - lr * Φ

    Note:
        W is modified in-place during the retraction step.
    """
    # Determine if TP is enabled
    ws, _ = _tp_world_and_rank(tp_group)
    tp_enabled = tp_group is not None and partition_dim is not None and ws > 1

    if not tp_enabled:
        # Single-rank path
        return _compute_single_rank(
            W=W,
            M=M,
            target_radius=target_radius,
            power_iteration_steps=power_iteration_steps,
            msign_steps=msign_steps,
            brent_tolerance_f=brent_tolerance_f,
            brent_tolerance_x=brent_tolerance_x,
            brent_max_iterations=brent_max_iterations,
        )
    else:
        # TP enabled: duplicated mode only
        if tp_mode != "duplicated":
            raise NotImplementedError(
                f"SpectralBall TP mode '{tp_mode}' not implemented; use 'duplicated' for now."
            )
        return _compute_tp_duplicated(
            W=W,
            M=M,
            target_radius=target_radius,
            power_iteration_steps=power_iteration_steps,
            msign_steps=msign_steps,
            brent_tolerance_f=brent_tolerance_f,
            brent_tolerance_x=brent_tolerance_x,
            brent_max_iterations=brent_max_iterations,
            tp_group=tp_group,
            partition_dim=partition_dim,
        )
