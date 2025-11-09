from itertools import chain, islice, repeat
import torch

our_coeffs_list = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
]

def deflate(abc, deflation_eps):
    a, b, c = abc
    return a / (1 + deflation_eps), b / (1 + deflation_eps)**3, c / (1 + deflation_eps)**5


@torch.compile
def polar_express(G: torch.Tensor, steps: int):
    assert G.ndim >= 2, "Input tensor must have at least two dimensions."
    assert steps > 0, "Number of steps must be positive."
    deflation_eps = 0.01
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):  # opposite convention from our other code
        X = X.mT
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1 + deflation_eps) + 1e-7)
    # NOTE: it's very important to make `hs` a plain list, not an iterator.
    # Don't do any CPU operations inside the loop, just GPU ops.
    # Otherwise it could seriously slow down the code.
    hs = [deflate(coefffs, deflation_eps) for coefffs in chain(
        islice((our_coeffs_list), steps),
        repeat(our_coeffs_list[-1], steps - len(our_coeffs_list)),
    )]

    for a, b, c in hs:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
  
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


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
