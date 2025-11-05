# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Matrix sign function implementation using Polar Express algorithm."""

import torch

# Pre-computed polynomial coefficients (a, b, c) for each iteration step
ABC_LIST: list[tuple[float, float, float]] = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
]

# Apply safe scaling to the first N-1 items (improves numerical stability), keep last item unchanged
ABC_LIST_STABLE: list[tuple[float, float, float]] = [
    (a / 1.01, b / (1.01**3), c / (1.01**5)) for (a, b, c) in ABC_LIST[:-1]
] + [ABC_LIST[-1]]


@torch.no_grad()
def msign(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Compute the matrix sign function using the Polar Express algorithm.
    Paper: https://arxiv.org/abs/2505.16932

    Functionality: Approximates sign(G) = G (G^T G)^{-1/2} for polar decomposition.

    Args:
        G: Input tensor, must have at least 2 dimensions.
        steps: Number of iteration steps for the algorithm.

    Returns:
        Matrix sign of G.
    """
    assert G.ndim >= 2, "Input tensor must have at least 2 dimensions."

    # If rows > cols, transpose for efficiency (processing tall matrices)
    should_transpose = G.size(-2) > G.size(-1)
    x = G.bfloat16()  # Use bfloat16 to save memory and accelerate
    if should_transpose:
        x = x.mT  # Batch transpose

    # Normalize: prevent numerical overflow, multiply by 1.01 as safety margin
    norm = x.norm(dim=(-2, -1), keepdim=True)
    x = x / (norm * 1.01)

    # Iterative update: use pre-computed polynomial coefficients
    for step in range(steps):
        # If step count exceeds preset coefficient count, reuse last coefficient
        if step < len(ABC_LIST_STABLE):
            a, b, c = ABC_LIST_STABLE[step]
        else:
            a, b, c = ABC_LIST_STABLE[-1]

        # Compute S = X X^T (symmetric Gram matrix)
        S = x @ x.mT

        # According to formula: X_{new} = (a I + b S + c S^2) X
        # To avoid explicitly constructing I, we directly manipulate the diagonal

        # First compute c * S
        Y = c * S

        # Add b to Y's diagonal → equivalent to b I + c S
        Y.diagonal(dim1=-2, dim2=-1).add_(b)

        # Then multiply by S → (b I + c S) S = b S + c S^2
        Y = Y @ S

        # Add a to the result's diagonal → a I + b S + c S^2
        Y.diagonal(dim1=-2, dim2=-1).add_(a)

        # Finally multiply by original X: X_new = (a I + b S + c S^2) X
        x = Y @ x

    # If previously transposed, transpose back
    if should_transpose:
        x = x.mT

    # Replace NaN/Inf with 0 (numerical safety fallback)
    x = torch.nan_to_num(x)

    # Return float32 (compatible with downstream)
    return x.float()


@torch.no_grad()
def msign_accurate(g: torch.Tensor) -> torch.Tensor:
    """
    Compute msign accurately using SVD (PyTorch version).
    Consistent with numpy version: msign(g) = U · sign(S) · Vh

    Args:
        g: Real or complex tensor of shape (..., m, n), supports batch SVD.

    Returns:
        Tensor of same shape as g.
    """
    # torch.linalg.svd supports batching, returns U, S, Vh
    # full_matrices=False maintains consistency with numpy version's economy SVD
    U, S, Vh = torch.linalg.svd(g, full_matrices=False)

    # Take sign of singular values. Note sign(0)=0, consistent with numpy
    S_sign = torch.sign(S)

    # Convert sign(S) to diagonal matrix, then do U @ diag @ Vh
    # For batch case, need to construct diagonal blocks
    # torch.diag_embed converts (..., k) -> (..., k, k)
    S_sign_diag = torch.diag_embed(S_sign)

    return U @ S_sign_diag @ Vh
