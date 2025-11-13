"""Utility functions for Spectral Ball optimizer."""

import math
from itertools import chain, islice, repeat
from typing import Optional, Tuple

import torch
from absl import logging


__all__ = [
    "compute_target_radius",
    "compute_spectral_ball_update",
    "solve_lambda_with_brent",
    "solve_lambda_with_bisection",
]


# =============================================================================
# Numerical Stability Checking Utilities
# =============================================================================
def _check_tensor(tensor: torch.Tensor, name: str, context: str = "") -> tuple:
    """
    Check tensor for numerical issues and return a concise log string.

    Returns:
        (msg, has_nan, has_inf)
    """
    try:
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()

        if has_nan or has_inf:
            # If unhealthy, get norm carefully
            try:
                norm_val = torch.linalg.norm(tensor).item()
            except:
                norm_val = float('nan')

            status = "✗"
            msg = f"[{context}] {status} {name}: norm={norm_val:.2e}, shape={tuple(tensor.shape)}, dtype={tensor.dtype}"
            if has_nan:
                msg += ", HAS_NAN=True"
            if has_inf:
                msg += ", HAS_INF=True"
        else:
            # Healthy tensor
            norm_val = torch.linalg.norm(tensor).item()
            msg = f"[{context}] ✓ {name}: norm={norm_val:.2e}, shape={tuple(tensor.shape)}, dtype={tensor.dtype}"

        return msg, has_nan, has_inf
    except Exception as e:
        return f"[{context}] ERROR checking {name}: {str(e)}", False, False


def _log_tensor(tensor: torch.Tensor, name: str, context: str = ""):
    """Check and log tensor health."""
    msg, has_nan, has_inf = _check_tensor(tensor, name, context)

    if has_nan or has_inf:
        logging.error(msg)
    else:
        logging.debug(msg)

    return has_nan or has_inf


def _muon_newton_schulz_step(X: torch.Tensor, a: float, b: float, c: float) -> torch.Tensor:
    """One Newton-Schulz iteration: X ← a·X + X·(b·A + c·A²) where A = X·X^T."""
    A = X @ X.mT
    B = torch.addmm(A, A, A, alpha=c, beta=b)
    X = torch.addmm(X, B, X, alpha=1.0, beta=a)
    return X

@torch.compile
def msign(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Matrix sign via Newton-Schulz with Polar-Express coefficients."""
    if G.ndim < 2:
        raise ValueError("Input tensor must have at least 2 dimensions.")
    if G.dtype != torch.float32:
        raise ValueError(f"Input tensor G must be in float32")

    transpose = G.size(-2) > G.size(-1)
    X = G.mT if transpose else G
    X = torch.nn.functional.normalize(X, p=2, dim=(-2, -1), eps=1e-7)
    X = X.to(torch.bfloat16)
    
    coeffs = [
        (8.2051, -22.9019, 16.4607),
        (4.0664, -2.8612, 0.5184),
        (3.9096, -2.8234, 0.5250),
        (3.2856, -2.4153, 0.4853),
        (2.2779, -1.6198, 0.3985),
        (1.8726, -1.2307, 0.3585),
        (1.8564, -1.2132, 0.3568),
        (1.8750, -1.2500, 0.3750),
    ]

    for i in range(steps):
        a, b, c = coeffs[i % 8]
        X = _muon_newton_schulz_step(X, a, b, c)

    return X.mT if transpose else X


@torch.no_grad()
def power_iteration(w: torch.Tensor, steps: int = 50, eps: float = 1e-20):
    """Leading singular triplet (σ, u, v) via bilateral power iteration (fp32)."""
    if w.ndim < 2:
        raise ValueError("Input tensor must have at least 2 dimensions.")

    # Check input
    w_norm = torch.linalg.norm(w).item()
    has_issue = _log_tensor(w, "w_input", "power_iteration")
    if has_issue:
        logging.error(f"[power_iteration] Input W has numerical issues! norm={w_norm:.6e}")

    w = w.to(torch.float32)
    v = torch.ones_like(w[..., :1, :].transpose(-2, -1))
    for _ in range(steps):
        v = torch.nn.functional.normalize(w.transpose(-2, -1) @ (w @ v), dim=-2)
    u = torch.nn.functional.normalize(w @ v, dim=-2)
    s = (u.transpose(-2, -1) @ w @ v).squeeze(-1).squeeze(-1)

    # Check outputs
    sigma_value = s.item()
    logging.warning(f"[power_iteration] sigma={sigma_value:.6e}, w_norm={w_norm:.6e}, shape={tuple(w.shape)}")

    # Check if u, v have issues
    _log_tensor(u, "u", "power_iteration")
    _log_tensor(v, "v", "power_iteration")

    # Warning if sigma is abnormal
    if sigma_value < 1e-6:
        logging.error(
            f"[power_iteration] ✗ sigma={sigma_value:.6e} is too small! "
            f"w_norm={w_norm:.6e}, w.shape={tuple(w.shape)}"
        )
    elif sigma_value > 1e6:
        logging.warning(f"[power_iteration] ⚠️ sigma={sigma_value:.6e} is very large!")

    return s, u, v


@torch.no_grad()
def inner_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Frobenius inner product <a, b>."""
    return (a * b).sum()


@torch.no_grad()
def compute_phi(G: torch.Tensor, Theta: torch.Tensor, lambda_value: float, msign_steps: int = 5) -> torch.Tensor:
    """Φ(λ) = msign(G + λΘ)."""
    z = G + lambda_value * Theta

    # Check Z before passing to msign
    has_z_issue = _log_tensor(z, f"Z_before_msign(lambda={lambda_value:.2e})", "compute_phi")
    if has_z_issue:
        logging.error(f"[compute_phi] Z has numerical issue BEFORE msign! lambda={lambda_value:.6e}")
        # Also check components
        _log_tensor(G, "G_in_compute_phi", "compute_phi")
        _log_tensor(Theta, "Theta_in_compute_phi", "compute_phi")
        lambda_Theta = lambda_value * Theta
        _log_tensor(lambda_Theta, f"lambda*Theta(lambda={lambda_value:.2e})", "compute_phi")

    Phi = msign(z, steps=msign_steps)

    # Check Phi after msign - if bad, investigate Z
    has_phi_issue = _log_tensor(Phi, f"Phi_after_msign(lambda={lambda_value:.2e})", "compute_phi")
    if has_phi_issue:
        logging.error(f"[compute_phi] ✗ msign returned bad Phi! lambda={lambda_value:.6e}")
        # Get Z statistics
        z_norm = torch.linalg.norm(z).item()
        z_max = z.abs().max().item()
        z_min = z.abs().min().item()
        logging.error(
            f"[compute_phi] Z stats: norm={z_norm:.6e}, abs_max={z_max:.6e}, abs_min={z_min:.6e}"
        )
        # Also log G and Theta
        _log_tensor(G, "G_in_compute_phi", "compute_phi")
        _log_tensor(Theta, "Theta_in_compute_phi", "compute_phi")

    return Phi


@torch.no_grad()
def compute_f(G: torch.Tensor, Theta: torch.Tensor, lambda_value: float, msign_steps: int = 8) -> float:
    """f(λ) = <Θ, msign(G + λΘ)>."""
    Phi = compute_phi(G, Theta, lambda_value, msign_steps)

    # Check Phi for numerical issues
    has_issue = _log_tensor(Phi, f"Phi_in_compute_f(lambda={lambda_value:.2e})", "compute_f")

    f_value = float(inner_product(Theta, Phi).item())

    # Log f value only if there's an issue or if f is nan/inf
    if has_issue or not math.isfinite(f_value):
        logging.error(f"[compute_f] lambda={lambda_value:.6e}, f={f_value:.6e}, Phi_has_issue={has_issue}")

    return f_value


@torch.no_grad()
def find_bracket(
    G: torch.Tensor,
    Theta: torch.Tensor,
    initial_guess: float = 0.0,
    initial_step: float = 1.0,
    max_expansions: int = 20,
    msign_steps: int = 8,
    tolerance_f: float = 1e-8,
) -> Tuple[float, float, float, float]:
    """
    Minimal-call bracketing search for a monotonic function f(λ) with a unique zero:
    1. Evaluate f₀ = f(λ₀)
    2. Expand exponentially in a single direction determined by f₀'s sign until a sign change occurs

    """
    f0 = compute_f(G, Theta, initial_guess, msign_steps)
    logging.debug(f"[find_bracket] Initial: f0={f0:.6e}")

    if abs(f0) < tolerance_f:
        logging.debug(f"[find_bracket] Converged at λ={initial_guess:.6f}, |f|={abs(f0):.6e}")
        return initial_guess, initial_guess, f0, f0
    
    return -1e-3, 1e-3, compute_f(G,Theta,-1e-3,msign_steps), compute_f(G,Theta,1e-3,msign_steps) #经验性的数值，lambda都在1e-3量级以下
    # direction = 1.0 if f0 < 0.0 else -1.0
    # step = initial_step if initial_step > 0.0 else 1.0
    # logging.debug(f"[find_bracket] Search direction: {'positive' if direction > 0 else 'negative'}, initial_step={step:.6f}")

    # a, fa = initial_guess, f0
    # b, fb = a, fa

    # for i in range(max_expansions):
    #     b = initial_guess + direction * step
    #     fb = compute_f(G, Theta, b, msign_steps)
    #     logging.debug(f"[find_bracket] Expansion {i+1}/{max_expansions}: λ={b:.6f}, f={fb:.6e}")

    #     if fa * fb <= 0.0 or abs(fb) < tolerance_f or abs(fa) < tolerance_f:
    #         if a > b:
    #             a, b, fa, fb = b, a, fb, fa
    #         logging.debug(
    #             f"[find_bracket] ✓ SUCCESS after {i+1} expansions: "
    #             f"bracket=[{a:.6f}, {b:.6f}], f(a)={fa:.6e}, f(b)={fb:.6e}, best_|f|={min(abs(fa), abs(fb)):.6e}"
    #         )
    #         return a, b, fa, fb

    #     a, fa = b, fb
    #     step *= 2.0

    # logging.warning(
    #     f"[find_bracket] ✗ FAILED after {max_expansions} expansions: "
    #     f"last_interval=[{a:.6f}, {b:.6f}], f(a)={fa:.6e}, f(b)={fb:.6e}. "
    #     f"No sign change (both {'positive' if fa > 0 and fb > 0 else 'negative' if fa < 0 and fb < 0 else 'mixed?'})"
    # )
    # return 0.0, 0.0, f0, f0


@torch.no_grad()
def solve_lambda_with_brent(
    G: torch.Tensor,
    Theta: torch.Tensor,
    initial_guess: float = 0.0,
    initial_step: float = 1.0,
    tolerance_f: float = 1e-8,
    max_iterations: int = 10,
    max_expansions: int = 10,
    msign_steps: int = 5,
) -> Tuple[float, bool, float, int]:
    """
    Solve λ such that f(λ) = <Θ, msign(G + λΘ)> ≈ 0 using Brent's method.
    The algorithm minimizes |f(λ)|; convergence is determined only by |f|, not |x|.
    """

    import math

    logging.debug(f"[brent] Starting solver: tolerance_f={tolerance_f:.2e}, max_iter={max_iterations}")

    # --- Step 1. Find bracket [a,b] ---
    a, b, fa, fb = find_bracket(
        G, Theta,
        initial_guess=initial_guess,
        initial_step=initial_step,
        max_expansions=max_expansions,
        msign_steps=msign_steps,
        tolerance_f=tolerance_f,
    )

    # --- Step 2. Check bracket validity ---
    if a == b:
        logging.warning(
            f"[brent] ✗ No valid bracket found. "
            f"Returning λ={a:.6f} with |f|={abs(fa):.6e} (iterations=0)"
        )
        return a, False, abs(fa), 0
    if fa > fb:  # ensure f(a) < 0 < f(b)
        a, b, fa, fb = b, a, fb, fa

    logging.debug(f"[brent] Starting bracket: [{a:.6f}, {b:.6f}], f(a)={fa:.6e}, f(b)={fb:.6e}")

    # --- Step 3. Initialize for Brent iterations ---
    c, fc = a, fa
    d = e = b - a
    best_lambda, best_f = b, fb
    logging.debug(f"[brent] Initial best: λ={best_lambda:.6f}, |f|={abs(best_f):.6e}")

    # --- Step 4. Brent loop ---
    for it in range(1, max_iterations + 1):
        # Stop if f is already small enough
        if abs(fb) <= tolerance_f:
            logging.debug(
                f"[brent] ✓ CONVERGED at iteration {it}: "
                f"λ={b:.6f}, |f|={abs(fb):.6e} <= {tolerance_f:.2e}"
            )
            return b, True, abs(fb), it

        # Maintain bracketing condition
        if fa * fb > 0:
            a, fa = c, fc
            c, fc = b, fb
            d = e = b - a
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa

        # Midpoint
        m = 0.5 * (c - b)

        # Update best (closest to zero)
        if abs(fb) < abs(best_f):
            best_lambda, best_f = b, fb
            logging.debug(f"[brent] Iter {it}: new best λ={best_lambda:.6f}, |f|={abs(best_f):.6e}")
        else:
            logging.debug(f"[brent] Iter {it}: λ={b:.6f}, f={fb:.6e} (not better)")

        # --- Simplified Brent step: secant/interpolation fallback ---
        # abs(e) > 1e-3 → previous step size not too small (not yet pure bisection)
        # abs(fa - fc) > 1e-4 → denominator for secant not nearly zero
        # If both hold, try interpolation; otherwise fall back to bisection.
        if abs(e) > 1e-3 and abs(fa - fc) > 1e-4:
            s = fb / fa
            p = 2.0 * m * s
            q = 1.0 - s
            if p > 0:
                q = -q
            p = abs(p)
            if 2.0 * p < abs(e * q):
                d, e = p / q, d
            else:
                d = e = m
        else:
            # Fall back to simple bisection step when interpolation is unsafe
            d = e = m

        # keep previous point for interpolation / bracketing
        c, fc = b, fb
        # take the step; if it's too tiny (stagnation), force a minimal move toward the midpoint direction
        b += d if abs(d) > tolerance_f else math.copysign(tolerance_f, m)
        # evaluate f at the new iterate
        fb = compute_f(G, Theta, b, msign_steps)


    # --- Step 5. Return best |f| ---
    logging.warning(
        f"[brent] ✗ NOT CONVERGED after {max_iterations} iterations: "
        f"best λ={best_lambda:.6f}, |f|={abs(best_f):.6e} (target: {tolerance_f:.2e})"
    )
    return best_lambda, False, abs(best_f), max_iterations




# =============================================================================
# Bisection Solver for Lagrange Multiplier
# =============================================================================
@torch.no_grad()
def solve_lambda_with_bisection(
    G: torch.Tensor,
    Theta: torch.Tensor,
    initial_guess: float = 0.0,
    initial_step: float = 1.0,
    tolerance_f: float = 1e-5,
    max_iterations: int = 10,
    max_expansions: int = 10,
    msign_steps: int = 8,
) -> Tuple[float, bool, float, int]:
    """
    Solve λ such that f(λ) = <Θ, msign(G + λΘ)> = 0 using bisection.
    Assumptions:
      - f(λ) is strictly monotonic increasing and has a unique root.
    Behavior:
      - Uses a monotone-aware bracketing routine `find_bracket` to obtain [a, b] with f(a) < 0 < f(b).
      - Tracks the best λ (smallest |f|) across iterations; if not converged by max_iterations,
        returns the best-seen λ.
    Returns:
      (lambda_value, converged, residual, iterations)
    """

    logging.debug(f"[bisection] Starting solver: tolerance_f={tolerance_f:.2e}, max_iter={max_iterations}")

    # 1) Bracket the root with opposite signs.
    a, b, fa, fb = find_bracket(
        G, Theta,
        initial_guess=initial_guess,
        initial_step=initial_step,
        max_expansions=max_expansions,
        msign_steps=msign_steps,
        tolerance_f=tolerance_f,
    )

    # Early exits if an endpoint already satisfies the tolerance
    if abs(fa) <= tolerance_f:
        logging.debug(f"[bisection] ✓ Converged at endpoint a: λ={a:.6f}, |f|={abs(fa):.6e}")
        return a, True, abs(fa), 0
    if abs(fb) <= tolerance_f:
        logging.debug(f"[bisection] ✓ Converged at endpoint b: λ={b:.6f}, |f|={abs(fb):.6e}")
        return b, True, abs(fb), 0

    # Initialize "best so far" (min |f|)
    best_lambda, best_f = (a, fa) if abs(fa) < abs(fb) else (b, fb)
    logging.debug(f"[bisection] Initial best: λ={best_lambda:.6f}, |f|={abs(best_f):.6e}")

    # 2) Bisection iterations
    for it in range(1, max_iterations + 1):
        mid = 0.5 * (a + b)
        f_mid = compute_f(G, Theta, mid, msign_steps)

        # Update best (closest to zero by absolute value)
        if abs(f_mid) < abs(best_f):
            best_lambda, best_f = mid, f_mid
            logging.debug(f"[bisection] Iter {it}: new best λ={best_lambda:.6f}, |f|={abs(best_f):.6e}")
        else:
            logging.debug(f"[bisection] Iter {it}: λ={mid:.6f}, f={f_mid:.6e} (not better)")

        # Converged by function-value tolerance
        if abs(f_mid) <= tolerance_f:
            logging.debug(
                f"[bisection] ✓ CONVERGED at iteration {it}: "
                f"λ={mid:.6f}, |f|={abs(f_mid):.6e} <= {tolerance_f:.2e}"
            )
            return mid, True, abs(f_mid), it

        # Monotone increasing: f_mid < 0 ⇒ root in [mid, b]; else in [a, mid]
        if f_mid < 0.0:
            a, fa = mid, f_mid
        else:
            b, fb = mid, f_mid

    # 3) Not converged within max_iterations: return best-so-far
    logging.warning(
        f"[bisection] ✗ NOT CONVERGED after {max_iterations} iterations: "
        f"best λ={best_lambda:.6f}, |f|={abs(best_f):.6e} (target: {tolerance_f:.2e})"
    )
    return best_lambda, False, abs(best_f), max_iterations



def compute_target_radius(shape: tuple, radius_mode: str, current_weight: Optional[torch.Tensor] = None) -> float:
    """Compute target radius R: 'spectral_mup' → sqrt(n_out/n_in), 'identity' → 1.0."""
    if radius_mode == "spectral_mup":
        n_out, n_in = shape
        return math.sqrt(n_out / n_in)
    elif radius_mode == "identity":
        return 1.0
    else:
        raise ValueError(f"Invalid radius_mode: {radius_mode}. Must be 'spectral_mup' or 'identity'.")


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
) -> torch.Tensor:
    """Compute spectral ball update for single-rank (non-TP) case.

    This implements the core algorithm:
    1. Power iteration to get σ, u, v
    2. Retract W to spectral sphere: W ← (R/σ)W
    3. Form Θ = uv^T
    4. Solve for λ: <Θ, msign(M + λΘ)> = 0
    5. Return Φ = msign(M + λΘ)
    """
    # Optional: only log on rank 0 in DDP
    # is_main_process = (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0)
    is_main_process = True  # set to False to silence

    # === Step 0: Check inputs (only log if issues) ===
    if is_main_process:
        has_w_issue = _log_tensor(W, "W_input", "SpectralBall")
        has_m_issue = _log_tensor(M, "M_input", "SpectralBall")
        if has_w_issue:
            logging.error(f"[SpectralBall] ✗ W_input has issues!")
        if has_m_issue:
            logging.error(f"[SpectralBall] ✗ M_input has issues!")

    # Convert M to fp32 once at the beginning
    M_fp32 = M.to(torch.float32)

    # Check M before normalization
    if is_main_process:
        M_norm_before = torch.linalg.norm(M_fp32, dim=(-2,-1), keepdim=True).item()
        # Only log if abnormal
        if M_norm_before < 1e-6 or M_norm_before > 1e3:
            logging.warning(f"[SpectralBall] ⚠️ M_norm_before_normalize abnormal: {M_norm_before:.6e}")

    M_fp32 = M_fp32 / (torch.linalg.norm(M_fp32, dim=(-2,-1), keepdim=True).clamp_min(1e-8))  # 归一化梯度

    # Check M after normalization only if has issues
    if is_main_process:
        has_m_issue = _log_tensor(M_fp32, "M_normalized", "SpectralBall")
        if has_m_issue:
            logging.error(f"[SpectralBall] ✗ M_normalized has issues!")

    # 1. Power iteration (returns fp32)
    sigma, u, v = power_iteration(W, steps=power_iteration_steps)
    sigma_value = sigma.item()

    # Check sigma value and compute scale_factor
    if is_main_process:
        # Warning if sigma is too small (scale_factor will be huge)
        if sigma_value < 1e-6:
            if sigma_value > 0:
                scale_factor_preview = target_radius / sigma_value
                logging.error(
                    f"[SpectralBall] ✗ sigma={sigma_value:.6e} is very small! "
                    f"scale_factor would be {scale_factor_preview:.2e}"
                )
            else:
                logging.error(
                    f"[SpectralBall] ✗ sigma={sigma_value:.6e} is zero or negative!"
                )

    # 2. Retract W to spectral sphere
    if sigma_value > 0:
        scale_factor = target_radius / (sigma_value+1e-8)

        # Warning if scale_factor is extreme
        if is_main_process and (scale_factor > 1e3 or scale_factor < 1e-3):
            logging.warning(
                f"[SpectralBall] ⚠️ Extreme scale_factor={scale_factor:.2e} "
                f"(sigma={sigma_value:.6e}, target_radius={target_radius:.6e})"
            )

        W.mul_(scale_factor)

        if is_main_process:
            # Only log retraction if scale_factor is extreme, otherwise silent
            if scale_factor > 1e3 or scale_factor < 1e-3:
                _log_tensor(W, "W_after_retraction", "SpectralBall")
                # Already logged in the extreme check above
    else:
        if is_main_process:
            logging.warning(f"[SpectralBall] ⚠️ Singular value sigma={sigma_value} <= 0, skipping retraction")

    # 3. Form Theta (fp32)
    Theta = u @ v.transpose(-2, -1)

    if is_main_process:
        has_theta_issue = _log_tensor(Theta, "Theta", "SpectralBall")
        if has_theta_issue:
            logging.error(f"[SpectralBall] ✗ Theta has numerical issues!")

        # Check for extreme values in M_fp32 and Theta - only log if abnormal
        M_max = M_fp32.abs().max().item()
        Theta_max = Theta.abs().max().item()

        # Warning if values are extreme (threshold: 1e2)
        if M_max > 1e2 or Theta_max > 1e2:
            M_min = M_fp32.abs().min().item()
            Theta_min = Theta.abs().min().item()
            logging.error(
                f"[SpectralBall] ✗ Extreme values: M_max={M_max:.2e}, M_min={M_min:.2e}, "
                f"Theta_max={Theta_max:.2e}, Theta_min={Theta_min:.2e}"
            )

    # 4. Solve for lambda using selected solver
    if solver == "bisection":
        lambda_value, converged, residual, iterations = solve_lambda_with_bisection(
            G=M_fp32,
            Theta=Theta,
            initial_guess=0.0,
            initial_step=1.0,
            tolerance_f=solver_tolerance_f,
            max_iterations=solver_max_iterations,
            max_expansions=10,
            msign_steps=msign_steps,
        )
    else:  # solver == "brent"
        lambda_value, converged, residual, iterations = solve_lambda_with_brent(
            G=M_fp32,
            Theta=Theta,
            initial_guess=0.0,
            initial_step=1.0,
            tolerance_f=solver_tolerance_f,
            max_iterations=solver_max_iterations,
            max_expansions=10,
            msign_steps=msign_steps,
        )
    if is_main_process:
        # Only log if not converged
        if not converged:
            logging.warning(
                f"[SpectralBall] ⚠️ Lambda NOT converged: lambda={lambda_value:.6e}, "
                f"residual={residual:.2e}, iters={iterations}"
            )

    # 5. Compute final update direction
    Z = M_fp32 + lambda_value * Theta

    if is_main_process:
        # Only log Z if it has issues
        has_z_issue = _log_tensor(Z, "Z", "SpectralBall")
        if has_z_issue:
            logging.error(f"[SpectralBall] ✗ Z has issues! lambda={lambda_value:.6e}")

    Phi = msign(Z, steps=msign_steps)

    if is_main_process:
        has_phi_issue = _log_tensor(Phi, "Phi_output", "SpectralBall")
        if has_phi_issue:
            logging.error(f"[SpectralBall] ✗ Final Phi has numerical issues! lambda={lambda_value:.6e}")

    return Phi


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
        solver: Solver method ('brent' or 'bisection')
        solver_tolerance_f: Function tolerance for solver
        solver_max_iterations: Maximum solver iterations
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
    M_full_fp32 = M_full_fp32 / (torch.linalg.norm(M_full_fp32, dim=(-2,-1), keepdim=True).clamp_min(1e-8))  # 归一化梯度

    # 1. Power iteration on global W (returns fp32)
    sigma, u, v = power_iteration(W_full, steps=power_iteration_steps)
    sigma_value = sigma.item()

    # 2. Retract global W and update local shard
    if sigma_value > 0:
        scale_factor = target_radius / (sigma_value + 1e-8)
        W_full_retracted = W_full * scale_factor
        # Split back to local shard and update original W
        W_local = _tp_split_along_dim(W_full_retracted, tp_group, partition_dim)
        W.copy_(W_local)
        logging.debug(
            f"[TP] Retracted W: sigma={sigma_value:.6f}, target={target_radius:.6f}, "
            f"scale={scale_factor:.6f}"
        )
    else:
        logging.debug(
            f"[TP] Singular value sigma={sigma_value} <= 0, skipping retraction"
        )

    # 3. Form Theta (fp32)
    Theta_full = u @ v.transpose(-2, -1)

    # 4. Solve for lambda on global tensors using selected solver
    if solver == "bisection":
        lambda_value, converged, residual, iterations = solve_lambda_with_bisection(
            G=M_full_fp32,
            Theta=Theta_full,
            initial_guess=0.0,
            initial_step=1.0,
            tolerance_f=solver_tolerance_f,
            max_iterations=solver_max_iterations,
            max_expansions=10,
            msign_steps=msign_steps,
        )
    else:  # solver == "brent"
        lambda_value, converged, residual, iterations = solve_lambda_with_brent(
            G=M_full_fp32,
            Theta=Theta_full,
            initial_guess=0.0,
            initial_step=1.0,
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
    return Phi_local


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
.

    Args:
        W: Current weight matrix (modified in-place for retraction)
        M: Momentum tensor
        target_radius: Target spectral norm R
        power_iteration_steps: Number of power iteration steps
        msign_steps: Number of Newton-Schulz iterations (uses Polar-Express coefficients)
        solver: Solver method ('brent' or 'bisection')
        solver_tolerance_f: Function tolerance for solver
        solver_max_iterations: Maximum solver iterations
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
            solver=solver,
            solver_tolerance_f=solver_tolerance_f,
            solver_max_iterations=solver_max_iterations,
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
        )
