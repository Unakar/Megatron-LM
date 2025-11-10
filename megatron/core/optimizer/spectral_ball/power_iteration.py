import torch


@torch.no_grad()
def power_iteration(w: torch.Tensor, steps: int = 10, eps: float = 1e-20):
    """Return leading singular triplet (σ, u, v) via bilateral power iteration.
    All tensors are float32 on the same device as w.
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


@torch.no_grad()
def top_svd(w: torch.Tensor):
    """Return leading singular triplet (σ, u, v) via torch.linalg.svd (dense)."""
    W = w.to(torch.float32)
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    u1 = U[:, :1]
    s1 = S[0]
    v1 = Vh[:1, :].transpose(-2, -1)
    return s1, u1, v1