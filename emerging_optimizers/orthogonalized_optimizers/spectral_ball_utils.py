"""Utility functions for Spectral Ball optimizer."""

import math
from itertools import chain, islice, repeat
from typing import Optional, Tuple

import torch
from absl import logging


DEBUG_CONVERGED = False 
DEBUG_NOT_CONVERGED = False 

__all__ = [
    "compute_target_radius",
    "compute_spectral_ball_update",
    "solve_lambda_with_bisection",
    # "solve_lambda_with_bisection_gpu",
]


# Polar-Express coefficients for Newton-Schulz iteration (8 steps max)
_MSIGN_COEFFS = (
    (8.2051, -22.9019, 16.4607),
    (4.0664, -2.8612, 0.5184),
    (3.9096, -2.8234, 0.5250),
    (3.2856, -2.4153, 0.4853),
    (2.2779, -1.6198, 0.3985),
    (1.8726, -1.2307, 0.3585),
    (1.8564, -1.2132, 0.3568),
    (1.8750, -1.2500, 0.3750),
)


def _msign_kernel(X: torch.Tensor, steps: int) -> torch.Tensor:
    """Core Newton-Schulz iteration kernel (fp32 only).
    
    Args:
        X: Normalized input tensor in fp32. Shape: [..., m, n] where m <= n.
        steps: Number of iterations (5 or 8).
    
    Returns:
        Matrix sign approximation in fp32.
    """
    for i in range(steps):
        a, b, c = _MSIGN_COEFFS[i]
        A = X @ X.mT
        B = torch.addmm(A, A, A, alpha=c, beta=b)
        X = torch.addmm(X, B, X, alpha=1.0, beta=a)
    return X


@torch.no_grad()
def msign(G: torch.Tensor, steps: int = 8) -> torch.Tensor:
    """Matrix sign via Newton-Schulz with Polar-Express coefficients.
    
    Args:
        G: Input tensor in fp32. Shape: [..., m, n]
        steps: Number of iterations (5 or 8).
    
    Returns:
        Matrix sign approximation, same shape as input.
    
    Warning:
        DO NOT use bf16! Newton-Schulz is extremely sensitive to rounding.
    """
    # For tall matrices (m > n), transpose to wide, compute, transpose back
    if G.size(-2) > G.size(-1):
        X = torch.nn.functional.normalize(G.mT, p=2, dim=(-2, -1), eps=1e-7)
        return _msign_kernel(X, steps).mT
    else:
        X = torch.nn.functional.normalize(G, p=2, dim=(-2, -1), eps=1e-7)
        return _msign_kernel(X, steps)


@torch.compile
def _power_iteration_kernel(
    w: torch.Tensor,
    u: torch.Tensor, 
    v: torch.Tensor,
    steps: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bilateral power iteration kernel (bf16, compiled).
    
    Args:
        w: Weight matrix in bf16. Shape: [m, n]
        u: Left singular vector in bf16. Shape: [m, 1]
        v: Right singular vector in bf16. Shape: [n, 1]
        steps: Number of iterations.
    
    Returns:
        Updated (u, v) in bf16.
    """
    wT = w.mT
    for _ in range(steps):
        v = torch.nn.functional.normalize(wT @ u, dim=0)
        u = torch.nn.functional.normalize(w @ v, dim=0)
    return u, v


@torch.no_grad()
def power_iteration(
    w: torch.Tensor,
    steps: int = 5,
    u_init: Optional[torch.Tensor] = None,
    v_init: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Leading singular triplet (σ, u, v) via bilateral power iteration.
    
    Uses alternating updates in bf16 for speed:
        u = normalize(W @ v)
        v = normalize(W^T @ u)
    
    Args:
        w: Weight matrix. Shape: [m, n]
        steps: Number of power iteration steps.
        u_init: Optional initial left singular vector for warm-start. Shape: [m, 1]
        v_init: Optional initial right singular vector for warm-start. Shape: [n, 1]
    
    Returns:
        Tuple of (sigma, u, v) where:
        - sigma: Leading singular value (scalar)
        - u: Left singular vector, shape [m, 1]
        - v: Right singular vector, shape [n, 1]
    """
    m, n = w.shape[-2], w.shape[-1]
    w_bf16 = w.to(torch.bfloat16)
    
    # Initialize u, v (outside compiled kernel for flexibility)
    if u_init is not None and v_init is not None:
        u = u_init.to(torch.bfloat16)
        v = v_init.to(torch.bfloat16)
    else:
        # Cold-start: random init is better than ones for convergence
        v = torch.randn(n, 1, dtype=torch.bfloat16, device=w.device)
        v = torch.nn.functional.normalize(v, dim=0)
        u = torch.nn.functional.normalize(w_bf16 @ v, dim=0)
    
    # Run compiled kernel
    u, v = _power_iteration_kernel(w_bf16, u, v, steps)
    
    # Compute sigma in fp32 for precision
    u_fp32 = u.to(torch.float32)
    v_fp32 = v.to(torch.float32)
    w_fp32 = w.to(torch.float32)
    sigma = (u_fp32.mT @ w_fp32 @ v_fp32).squeeze()

    return sigma, u_fp32, v_fp32


@torch.no_grad()
def apply_retract(
    W: torch.Tensor,
    sigma: float,
    target_radius: float,
    mode: str = 'hard',
    alpha: float = 0.05,
    current_lr: Optional[float] = None,
) -> float:
    """Apply retraction to spectral sphere.

    Args:
        W: Weight matrix (modified in-place)
        sigma: Current spectral norm
        target_radius: Target radius R
        mode: 'hard' or 'dynamic'
        alpha: Step size for dynamic mode (ignored for hard mode)
        current_lr: Current learning rate (only used in dynamic mode to scale alpha)

    Returns:
        bias: The bias value used (only relevant for dynamic mode, 0.0 for hard mode)
    """
    if mode == 'hard':
        # Hard retraction: if sigma != R, scale W to have norm R
        if max(sigma, 0.0) + 1e-8 != target_radius:
            scale_factor = target_radius / (max(sigma, 0.0) + 1e-8)
            W.mul_(scale_factor)
        return 0.0

    elif mode == 'dynamic':
        # Dynamic retraction: bias = -sign(sigma - R), W *= (1 + alpha * current_lr * bias)
        # This aligns the retraction strength with weight decay: both scale with lr
        bias = -1.0 if sigma > target_radius else 1.0

        # If current_lr is provided, scale alpha by lr (to align with weight decay)
        # Otherwise, use alpha directly (backward compatibility)
        if current_lr is not None:
            effective_alpha = alpha * current_lr
        else:
            effective_alpha = alpha

        W.mul_(1.0 + effective_alpha * bias)
        return bias

    else:
        raise ValueError(f"Unknown retract mode: {mode}")


@torch.no_grad()
def inner_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Frobenius inner product <a, b>."""
    return (a * b).sum()


@torch.no_grad()
def compute_phi(G: torch.Tensor, Theta: torch.Tensor, lambda_value: float, msign_steps: int = 5) -> torch.Tensor:
    """Φ(λ) = msign(G + λΘ)."""
    z = G + lambda_value * Theta
    Phi = msign(z, steps=msign_steps)
    return Phi

@torch.no_grad()
def compute_f(G: torch.Tensor, Theta: torch.Tensor, lambda_value: float, msign_steps: int = 8) -> float:
    """f(λ) = <Θ, msign(G + λΘ)>. Returns scalar float (triggers GPU sync)."""
    Phi = compute_phi(G, Theta, lambda_value, msign_steps)
    f_value = float(inner_product(Theta, Phi).item())
    return f_value

@torch.compile
@torch.no_grad()
def compute_f_tensor(G: torch.Tensor, Theta: torch.Tensor, lambda_value: torch.Tensor, msign_steps: int = 8) -> torch.Tensor:
    """f(λ) = <Θ, msign(G + λΘ)>. Returns 0-d tensor (no GPU sync)."""
    z = G + lambda_value * Theta
    Phi = msign(z, steps=msign_steps)
    return inner_product(Theta, Phi)


# =============================================================================
# GPU-accelerated Lambda Solver
# =============================================================================
# Design: Exploits f(λ) = <Θ, msign(G + λΘ)> being strictly monotone increasing.
# 
# Algorithm:
# 1. Start at λ=0, compute f(0)
# 2. If |f(0)| < tol → done (root at 0)
# 3. Based on sign of f(0), search in one direction:
#    - f(0) < 0 → root is to the right (λ > 0), step = +initial_step
#    - f(0) > 0 → root is to the left (λ < 0), step = -initial_step
# 4. Exponential expansion until sign change found
# 5. Once bracket found, use Illinois method for fast convergence
# =============================================================================

@torch.compile
@torch.no_grad()
def _gpu_illinois_refine(
    G: torch.Tensor,
    Theta: torch.Tensor,
    lambda_L: torch.Tensor,
    lambda_R: torch.Tensor,
    f_L: torch.Tensor,
    f_R: torch.Tensor,
    msign_steps: int,
    max_iterations: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GPU-compiled Illinois refinement (ZERO GPU-CPU sync).
    
    Given a valid bracket [λ_L, λ_R] with f_L <= 0 <= f_R, refines to find root.
    Uses Regula Falsi with Illinois modification to prevent stalling.
    
    Entire solver is compiled; msign kernel is NOT separately compiled
    to avoid nested compilation issues.
    
    All operations are tensor ops with torch.where for branching.
    Fixed iteration count - no early exit to avoid sync.
    
    Args:
        G: Normalized momentum (fp32)
        Theta: Rank-1 constraint u @ v^T (fp32)
        lambda_L, lambda_R: Bracket endpoints (0-d tensors)
        f_L, f_R: Function values at endpoints (0-d tensors)
        msign_steps: Number of msign iterations
        max_iterations: Number of Illinois iterations (fixed)
    
    Returns:
        (lambda_star, f_star): Best estimate and its function value (0-d tensors)
    """
    for _ in range(max_iterations):
        # Regula Falsi interpolation: λ_mid = λ_L - f_L * (λ_R - λ_L) / (f_R - f_L)
        lambda_mid = lambda_L - f_L * (lambda_R - lambda_L) / (f_R - f_L)
        
        # Compute f(λ_mid)
        z = G + lambda_mid * Theta
        Phi = msign(z, steps=msign_steps)
        f_mid = (Theta * Phi).sum()
        
        # Illinois update (all torch.where, no Python if):
        # f monotone increasing → f_L < 0 < f_R
        # f_mid < 0 → root in (mid, R) → update L, halve f_R weight
        # f_mid > 0 → root in (L, mid) → update R, halve f_L weight
        update_L = f_mid < 0
        
        lambda_L = torch.where(update_L, lambda_mid, lambda_L)
        lambda_R = torch.where(update_L, lambda_R, lambda_mid)
        f_L = torch.where(update_L, f_mid, f_L * 0.5)
        f_R = torch.where(update_L, f_R * 0.5, f_mid)
    
    # Final interpolation
    lambda_star = lambda_L - f_L * (lambda_R - lambda_L) / (f_R - f_L)
    z = G + lambda_star * Theta
    Phi = msign(z, steps=msign_steps)
    f_star = (Theta * Phi).sum()
    
    return lambda_star, f_star


@torch.no_grad()
def solve_lambda_with_bisection_gpu(
    G: torch.Tensor,
    Theta: torch.Tensor,
    initial_guess: float = 0.0,
    initial_step: float = 1e-3,
    tolerance_f: float = 1e-6,
    max_iterations: int = 20,
    max_expansions: int = 10,
    msign_steps: int = 8,
) -> Tuple[float, bool, float, int]:
    """GPU-accelerated λ solver. Interface matches solve_lambda_with_bisection.
    
    Design:
    - Bracket phase: Reuses find_bracket (accepts CPU sync for robustness)
    - Refine phase: GPU-compiled Illinois method (ZERO sync, fixed iterations)
    
    The main speedup comes from the refine phase: instead of calling msign
    and syncing each iteration, we compile the entire loop into one kernel.
    
    Args:
        G: Normalized momentum tensor (fp32)
        Theta: Rank-1 constraint matrix u @ v^T (fp32)
        initial_guess: Starting λ value
        initial_step: Initial step for bracket search
        tolerance_f: Convergence tolerance (used for bracket, not refine)
        max_iterations: Number of Illinois iterations
        max_expansions: Max bracket expansions
        msign_steps: Newton-Schulz iterations
    
    Returns:
        (lambda_star, converged, |f(lambda_star)|, iterations)
        Same interface as solve_lambda_with_bisection.
    """
    device = G.device
    dtype = G.dtype
    
    # === Phase 1: Find bracket (reuse existing, accepts CPU sync) ===
    λ_L, λ_R, f_L, f_R = find_bracket(
        G, Theta,
        initial_guess=initial_guess,
        initial_step=initial_step,
        max_expansions=max_expansions,
        msign_steps=msign_steps,
        tolerance_f=tolerance_f,
    )
    
    # Bracket failed → fallback to λ=0
    if λ_L is None:
        return 0.0, False, abs(f_L), 0
    
    # Degenerate bracket (already converged)
    if λ_L == λ_R:
        return float(λ_L), True, abs(f_L), 0
    
    # Check if bracket endpoint already satisfies tolerance (skip refine)
    if abs(f_L) < abs(f_R):
        best_λ, best_f = λ_L, f_L
    else:
        best_λ, best_f = λ_R, f_R
    
    if abs(best_f) <= tolerance_f:
        return float(best_λ), True, abs(best_f), 0
    
    # === Phase 2: GPU Illinois refinement (ZERO sync) ===
    # Convert bracket to tensors
    lambda_L_t = torch.tensor(λ_L, dtype=dtype, device=device)
    lambda_R_t = torch.tensor(λ_R, dtype=dtype, device=device)
    f_L_t = torch.tensor(f_L, dtype=dtype, device=device) if not isinstance(f_L, torch.Tensor) else f_L.to(dtype)
    f_R_t = torch.tensor(f_R, dtype=dtype, device=device) if not isinstance(f_R, torch.Tensor) else f_R.to(dtype)
    
    # Run compiled Illinois refinement
    lambda_star_t, f_star_t = _gpu_illinois_refine(
        G, Theta,
        lambda_L_t, lambda_R_t, f_L_t, f_R_t,
        msign_steps, max_iterations
    )
    
    # === Phase 3: Extract results (single sync at the end) ===
    lambda_star = lambda_star_t.item()
    f_star_abs = abs(f_star_t.item())
    converged = f_star_abs < tolerance_f
    
    return lambda_star, converged, f_star_abs, max_iterations


@torch.no_grad()
def find_bracket(
    G: torch.Tensor,
    Theta: torch.Tensor,
    initial_guess: float = 0.0,
    initial_step: float = 1e-3,
    max_expansions: int = 10,
    msign_steps: int = 8,
    tolerance_f: float = 1e-8,
) -> Tuple[float, float, float, float]:
    """
    Find λ_L < λ_R such that:
        f(λ_L) <= 0 <= f(λ_R)
    with f monotone increasing.

    If f(initial_guess) is already near zero, returns a degenerate bracket.
    Otherwise expands exponentially in the direction indicated by f0.
    """

    # Function handle
    f = compute_f_tensor

    # Initial λ and f
    λ0 = initial_guess
    f0 = f(G, Theta, λ0, msign_steps)

    # If already close to zero → return degenerate bracket
    if abs(f0) < tolerance_f:
        return λ0, λ0, f0, f0

    # Decide direction:
    #   f0 < 0 → root is to the right  → step > 0
    #   f0 > 0 → root is to the left   → step < 0
    step = initial_step if f0 < 0 else -initial_step

    λ_prev = λ0
    f_prev = f0

    for _ in range(max_expansions):

        λ_new = λ_prev + step
        f_new = f(G, Theta, λ_new, msign_steps)

        # ---------------------------
        # Check sign change:
        # f_prev ≤ 0 ≤ f_new OR f_new ≤ 0 ≤ f_prev
        # ---------------------------
        sign_prev = f_prev <= 0.0
        sign_new  = f_new  <= 0.0

        if sign_prev != sign_new:  # sign change occurred
            # ------------------------------------------------
            # Choose λ_L, λ_R based *on f*, NOT λ ordering.
            # Always enforce: f_L <= 0 <= f_R
            # ------------------------------------------------
            if f_prev <= 0 and f_new >= 0:
                λ_L, f_L = λ_prev, f_prev
                λ_R, f_R = λ_new, f_new
            elif f_new <= 0 and f_prev >= 0:
                λ_L, f_L = λ_new, f_new
                λ_R, f_R = λ_prev, f_prev
            else:
                # One point is extremely close to zero
                if abs(f_prev) <= abs(f_new):
                    λ_L = λ_R = λ_prev
                    f_L = f_R = f_prev
                else:
                    λ_L = λ_R = λ_new
                    f_L = f_R = f_new
            if DEBUG_CONVERGED:
                logging.warning(
                    f"[find_bracket] CONVERGED after {_ + 1} expansions. "
                    f"λ_L={λ_L:.6e}, f_L={f_L:.6e}, λ_R={λ_R:.6e}, f_R={f_R:.6e}."
                )
            return λ_L, λ_R, f_L, f_R

        # ------------------------------------------------
        # No sign change → expand search region
        # ------------------------------------------------
        step *= 2.0
        λ_prev, f_prev = λ_new, f_new

    # Failsafe
    logging.warning(
        f"[find_bracket] Could not bracket the root after {max_expansions} expansions. "
        f"Last λ={λ_prev:.6e}, f={f_prev:.6e}, w shape={G.shape}"
    )

    return None, None, f0, f0 #没找到，则区间返回none,直接返回f0




@torch.no_grad()
def solve_lambda_with_bisection(
    G: torch.Tensor,
    Theta: torch.Tensor,
    initial_guess: float = 0.0,
    initial_step: float = 1e-3,
    tolerance_f: float = 1e-6,
    max_iterations: int = 20,
    max_expansions: int = 10,
    msign_steps: int = 8,
) -> Tuple[float, bool, float, int]:
    """
    Solve λ such that f(λ) = <Θ, msign(G + λΘ)> = 0 using bisection.
    Assumes f is strictly monotone increasing.

    Returns:
        (lambda_star, converged_bool, |f(lambda_star)|, iterations_used)
    """

    # ----------------------------------------------------------------------
    # 1. Bracket the root: must satisfy f_L <= 0 <= f_R
    # ----------------------------------------------------------------------
    λ_L, λ_R, f_L, f_R = find_bracket(
        G, Theta,
        initial_guess=initial_guess,
        initial_step=initial_step,
        max_expansions=max_expansions,
        msign_steps=msign_steps,
        tolerance_f=tolerance_f,
    )

    # Bracketing failed
    if λ_L is None:
        logging.error("[bisect] find_bracket failed: cannot continue bisection.")
        return 0.0, False, f_L , 0 #其实就是直接返回lambda=0,退化为muon更新

    # ----------------------------------------------------------------------
    # 2. Pick best endpoint first 
    # ----------------------------------------------------------------------
    if abs(f_L) < abs(f_R):
        best_λ, best_f = λ_L, f_L
    else:
        best_λ, best_f = λ_R, f_R

    # If best endpoint already satisfies tolerance → done
    if abs(best_f) <= tolerance_f:
        if DEBUG_CONVERGED:
            logging.warning(
                f"[bisect] CONVERGED after bracketing search. "
                f"best λ={best_λ:.6e}, |f|={abs(best_f):.6e}."
            )
        return best_λ, True, abs(best_f), 0

    # ----------------------------------------------------------------------
    # 3. Standard monotone bisection
    # ----------------------------------------------------------------------
    for it in range(1, max_iterations + 1):

        λ_mid = 0.5 * (λ_L + λ_R)
        f_mid = compute_f_tensor(G, Theta, λ_mid, msign_steps)

        # Track best point (fallback)
        if abs(f_mid) < abs(best_f):
            best_λ, best_f = λ_mid, f_mid

        # Converged
        if abs(f_mid) <= tolerance_f:
            if DEBUG_CONVERGED:
                logging.warning(
                    f"[bisect] CONVERGED after {it} iterations. "
                    f"λ_mid={λ_mid:.6e}, |f|={abs(f_mid):.6e}."
                )
            return λ_mid, True, abs(f_mid), it

        # f is strictly increasing:
        # f_mid < 0 → root is in (mid, R)
        # f_mid > 0 → root is in (L, mid)
        if f_mid < 0:
            λ_L, f_L = λ_mid, f_mid
        else:
            λ_R, f_R = λ_mid, f_mid

    # ----------------------------------------------------------------------
    # 4. Not converged: return best-so-far
    # ----------------------------------------------------------------------
    if DEBUG_NOT_CONVERGED:
        logging.warning(
            f"[bisect] NOT CONVERGED after bisection search. "
            f"λ_L={λ_L:.6e}, f_L={f_L:.6e}, λ_R={λ_R:.6e}, f_R={f_R:.6e}."
        )
    return best_λ, False, abs(best_f), max_iterations


def compute_target_radius(shape: tuple, radius_mode: str, current_weight: Optional[torch.Tensor] = None, radius_scaler: float = 1.0) -> float:
    """Compute target radius R: 'spectral_mup' → sqrt(n_out/n_in) * scaler, 'identity' → 1.0 * scaler."""
    if radius_mode == "spectral_mup":
        n_out, n_in = shape
        return radius_scaler * math.sqrt(n_out / n_in)
    elif radius_mode == "identity":
        return radius_scaler * 1.0
    else:
        raise ValueError(f"Invalid radius_mode: {radius_mode}. Must be 'spectral_mup' or 'identity'.")

def get_spectral_ball_scale_factor(size_out: int, size_in: int, mode: str = "spectral", radius_scaler: float = 1.0) -> float:
    """Get the scale factor for the spectral ball update.

    This function mirrors Muon's scale factor to enable learning rate transferability.
    The default "align_adamw_rms" mode uses the same scaling as Muon for consistency.

    Args:
        size_out: The size of the output dimension (rows).
        size_in: The size of the input dimension (columns).
        mode: The mode to use for the scale.
            - "align_adamw_rms": 0.2 * max(size_out, size_in) ** 0.5 (default, matches Muon)
            - "shape_scaling": max(1, size_out / size_in) ** 0.5
            - "spectral_mup": radius_scaler * (size_out / size_in) ** 0.5
        radius_scaler: Scale factor for spectral_mup mode (default: 1.0).

    Returns:
        The scale factor for the update.
    """
    if mode == "shape_scaling":
        return max(1, size_out / size_in) ** 0.5
    elif mode == "align_adamw_rms":
        return 0.2 * max(size_out, size_in) ** 0.5
    elif mode == "spectral_mup":
        return radius_scaler * (size_out / size_in) ** 0.5
    else:
        raise ValueError(f"Invalid mode for SpectralBall update scale factor: {mode}")

@torch.no_grad()
def _tp_world_and_rank(tp_group: torch.distributed.ProcessGroup | None) -> tuple[int, int]:
    """Return (world_size, rank) from tp_group."""
    if tp_group is None:
        return 1, 0
    return tp_group.size(), tp_group.rank()


@torch.no_grad()
def _tp_gather_along_dim(x: torch.Tensor, tp_group: torch.distributed.ProcessGroup, dim: int) -> torch.Tensor:
    """All-gather shards along dim."""
    ws, _ = _tp_world_and_rank(tp_group)
    if ws == 1:
        return x
    shards = [torch.empty_like(x) for _ in range(ws)]
    torch.distributed.all_gather(shards, x, group=tp_group)
    return torch.cat(shards, dim=dim)


@torch.no_grad()
def _tp_split_along_dim(x_full: torch.Tensor, tp_group: torch.distributed.ProcessGroup, dim: int) -> torch.Tensor:
    """Split global tensor along dim, return local shard."""
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
    solver: str,
    solver_tolerance_f: float,
    solver_max_iterations: int,
    retract_mode: str = 'hard',
    retract_alpha: float = 0.05,
    current_lr: Optional[float] = None,
    use_gpu_bisection: bool = False,
    u_init: Optional[torch.Tensor] = None,
    v_init: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float, float, torch.Tensor, torch.Tensor]:
    """Compute spectral ball update for single-rank (non-TP) case.

    This implements the core algorithm:
    1. Power iteration to get σ, u, v
    2. Retract W to spectral sphere: W ← (R/σ)W
    3. Form Θ = uv^T
    4. Solve for λ: <Θ, msign(M + λΘ)> = 0
    5. Return Φ = msign(M + λΘ)

    Args:
        W: Current weight matrix (modified in-place for retraction)
        M: Momentum tensor
        target_radius: Target spectral norm R
        power_iteration_steps: Number of power iteration steps
        msign_steps: Number of Newton-Schulz iterations
        solver: Solver method ('bisection')
        solver_tolerance_f: Function tolerance for solver
        solver_max_iterations: Maximum solver iterations
        retract_mode: 'hard' or 'dynamic'
        retract_alpha: Step size for dynamic mode
        current_lr: Current learning rate (for dynamic retraction)
        use_gpu_bisection: Whether to use GPU bisection (unused)
        u_init: Optional initial left singular vector for warm-start
        v_init: Optional initial right singular vector for warm-start

    Returns:
        Tuple of (Phi, retract_bias, sigma_value, u, v) where:
        - Phi: Update direction
        - retract_bias: 0.0 for hard mode, ±1.0 for dynamic mode
        - sigma_value: Current spectral norm
        - u: Left singular vector (for caching)
        - v: Right singular vector (for caching)
    """

    # Convert M to fp32 once at the beginning
    M_fp32 = M.to(torch.float32)
    M_fp32 = M_fp32 / (torch.linalg.norm(M_fp32, dim=(-2,-1), keepdim=True).clamp_min(1e-8))  # 归一化梯度

    # 1. Power iteration (returns fp32), with optional warm-start
    sigma, u, v = power_iteration(W, steps=power_iteration_steps, u_init=u_init, v_init=v_init)
    sigma_value = sigma.item()

    # 2. Retract W to spectral sphere
    retract_bias = apply_retract(W, sigma_value, target_radius, mode=retract_mode, alpha=retract_alpha, current_lr=current_lr)


    # 3. Form Theta (fp32)
    Theta = u @ v.transpose(-2, -1)


    # 4. Solve for lambda using selected solver
    if solver == "bisection":
        if use_gpu_bisection:
            # GPU-accelerated solver (Illinois method, bracket on CPU, refine on GPU)
            lambda_value, converged, residual, iterations = solve_lambda_with_bisection_gpu(
                G=M_fp32,
                Theta=Theta,
                initial_guess=0.0,
                initial_step=1e-3,
                tolerance_f=solver_tolerance_f,
                max_iterations=solver_max_iterations,
                max_expansions=10,
                msign_steps=msign_steps,
            )
        else:
            # CPU solver (standard bisection)
            lambda_value, converged, residual, iterations = solve_lambda_with_bisection(
                G=M_fp32,
                Theta=Theta,
                initial_guess=0.0,
                initial_step=1e-3,
                tolerance_f=solver_tolerance_f,
                max_iterations=solver_max_iterations,
                max_expansions=10,
                msign_steps=msign_steps,
            )

    # 5. Compute final update direction
    Z = M_fp32 + lambda_value * Theta

    Phi = msign(Z, steps=msign_steps)

    return Phi, retract_bias, sigma_value, u, v


def _compute_tp_duplicated(
    W: torch.Tensor,
    M: torch.Tensor,
    target_radius: float,
    power_iteration_steps: int,
    msign_steps: int,
    solver: str,
    solver_tolerance_f: float,
    solver_max_iterations: int,
    tp_group: torch.distributed.ProcessGroup,
    partition_dim: int,
    retract_mode: str = 'hard',
    retract_alpha: float = 0.05,
    current_lr: Optional[float] = None,
    use_gpu_bisection: bool = False,
    u_init: Optional[torch.Tensor] = None,
    v_init: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float, float, torch.Tensor, torch.Tensor]:
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
        solver: Solver method ('bisection')
        solver_tolerance_f: Function tolerance for solver
        solver_max_iterations: Maximum solver iterations
        tp_group: Tensor parallel process group
        partition_dim: Dimension along which tensors are partitioned
        u_init: Optional initial left singular vector for warm-start
        v_init: Optional initial right singular vector for warm-start

    Returns:
        Tuple of (Phi_local, retract_bias, sigma_value, u, v)
    """
    # Gather shards to global matrices
    W_full = _tp_gather_along_dim(W, tp_group, partition_dim)
    M_full = _tp_gather_along_dim(M, tp_group, partition_dim)

    # Convert M to fp32 once
    M_full_fp32 = M_full.to(torch.float32)
    M_full_fp32 = M_full_fp32 / (torch.linalg.norm(M_full_fp32, dim=(-2,-1), keepdim=True).clamp_min(1e-8))  # 归一化梯度

    # 1. Power iteration on global W (returns fp32), with optional warm-start
    sigma, u, v = power_iteration(W_full, steps=power_iteration_steps, u_init=u_init, v_init=v_init)
    sigma_value = sigma.item()

    # 2. Retract global W and update local shard
    retract_bias = apply_retract(W_full, sigma_value, target_radius, mode=retract_mode, alpha=retract_alpha, current_lr=current_lr)
    # Split back to local shard and update original W
    W_local = _tp_split_along_dim(W_full, tp_group, partition_dim)
    W.copy_(W_local)

    # 3. Form Theta (fp32)
    Theta_full = u @ v.transpose(-2, -1)

    # 4. Solve for lambda on global tensors using selected solver
    if solver == "bisection":
        bisection_fn = solve_lambda_with_bisection
        lambda_value, converged, residual, iterations = bisection_fn(
            G=M_full_fp32,
            Theta=Theta_full,
            initial_guess=0.0,
            initial_step=1e-3,
            tolerance_f=solver_tolerance_f,
            max_iterations=solver_max_iterations,
            max_expansions=10,
            msign_steps=msign_steps,
        )
    if not converged:
        logging.warning(
            f"[TP] {solver.capitalize()} solver did not converge: residual={residual:.2e} "
            f"after {iterations} iterations"
        )

    # 5. Compute Φ on global tensor (no communication)
    Z_full = M_full_fp32 + lambda_value * Theta_full
    Phi_full = msign(Z_full, steps=msign_steps)

    # 6. Split back to local shard
    Phi_local = _tp_split_along_dim(Phi_full, tp_group, partition_dim)
    return Phi_local, retract_bias, sigma_value, u, v


def compute_spectral_ball_update(
    W: torch.Tensor,
    M: torch.Tensor,
    target_radius: float,
    power_iteration_steps: int,
    msign_steps: int,
    solver: str,
    solver_tolerance_f: float,
    solver_max_iterations: int,
    *,
    tp_group: torch.distributed.ProcessGroup | None = None,
    partition_dim: int | None = None,
    tp_mode: str = "duplicated",
    retract_mode: str = 'hard',
    retract_alpha: float = 0.05,
    current_lr: Optional[float] = None,
    use_gpu_bisection: bool = False,
    u_init: Optional[torch.Tensor] = None,
    v_init: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float, float, torch.Tensor, torch.Tensor]:
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

    Args:
        W: Current weight matrix (modified in-place for retraction)
        M: Momentum tensor
        target_radius: Target spectral norm R
        power_iteration_steps: Number of power iteration steps
        msign_steps: Number of Newton-Schulz iterations (uses Polar-Express coefficients)
        solver: Solver method ('bisection')
        solver_tolerance_f: Function tolerance for solver
        solver_max_iterations: Maximum solver iterations
        tp_group: Tensor parallel process group (None for single-rank)
        partition_dim: Dimension along which tensors are partitioned
        tp_mode: TP mode (only "duplicated" is currently supported)
        current_lr: Current learning rate (for dynamic retraction)
        u_init: Optional initial left singular vector for warm-start power iteration
        v_init: Optional initial right singular vector for warm-start power iteration

    Returns:
        Tuple of (Phi, retract_bias, sigma, u, v) where:
        - Phi: Update direction to be applied as W ← W - lr * Φ
        - retract_bias: Retraction bias (0.0 for hard mode)
        - sigma: Current spectral norm
        - u: Left singular vector (for caching)
        - v: Right singular vector (for caching)

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
            solver=solver,
            solver_tolerance_f=solver_tolerance_f,
            solver_max_iterations=solver_max_iterations,
            retract_mode=retract_mode,
            retract_alpha=retract_alpha,
            current_lr=current_lr,
            use_gpu_bisection=use_gpu_bisection,
            u_init=u_init,
            v_init=v_init,
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
            solver=solver,
            solver_tolerance_f=solver_tolerance_f,
            solver_max_iterations=solver_max_iterations,
            tp_group=tp_group,
            partition_dim=partition_dim,
            retract_mode=retract_mode,
            retract_alpha=retract_alpha,
            current_lr=current_lr,
            use_gpu_bisection=use_gpu_bisection,
            u_init=u_init,
            v_init=v_init,
        )