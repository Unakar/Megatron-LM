"""Benchmark script comparing PyTorch native vs Triton implementation of msign.

Compares:
1. PyTorch native: torch.addmm with X @ X.T
2. PyTorch compiled: torch.compile with reduce-overhead mode
3. Triton: newton_schulz with use_syrk=True (tsyrk kernel)

Usage:
    python benchmark_msign.py
"""

import sys
sys.path.insert(0, '/root/yangwang/Megatron-LM')

import torch
import time

# ============================================================================
# PyTorch Native Implementation (from spectral_ball_utils.py comments)
# ============================================================================

def _newton_schulz_step(X: torch.Tensor, a: float, b: float, c: float) -> torch.Tensor:
    """One Newton-Schulz iteration: X ← a·X + X·(b·A + c·A²) where A = X·X^T."""
    A = X @ X.mT
    B = torch.addmm(A, A, A, alpha=c, beta=b)
    X = torch.addmm(X, B, X, alpha=1.0, beta=a)
    return X


MSIGN_COEFFS = [
    (8.2051, -22.9019, 16.4607),
    (4.0664, -2.8612, 0.5184),
    (3.9096, -2.8234, 0.5250),
    (3.2856, -2.4153, 0.4853),
    (2.2779, -1.6198, 0.3985),
    (1.8726, -1.2307, 0.3585),
    (1.8564, -1.2132, 0.3568),
    (1.8750, -1.2500, 0.3750),
]


def _msign_inner(X: torch.Tensor) -> torch.Tensor:
    """Inner loop of msign - 8 Newton-Schulz iterations."""
    for i in range(8):
        a, b, c = MSIGN_COEFFS[i]
        X = _newton_schulz_step(X, a, b, c)
    return X


def msign_pytorch(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Matrix sign via Newton-Schulz with PyTorch native implementation."""
    if G.ndim < 2:
        raise ValueError("Input tensor must have at least 2 dimensions.")
    if G.dtype != torch.float32:
        raise ValueError(f"Input tensor G must be in float32")

    transpose = G.size(-2) > G.size(-1)
    X = G.mT if transpose else G
    X = torch.nn.functional.normalize(X, p=2, dim=(-2, -1), eps=1e-7)
    
    X = _msign_inner(X)

    return X.mT if transpose else X


# Compile the entire iteration loop, not just a single step
# This allows the compiler to optimize across all 8 iterations
@torch.compile(mode="default", dynamic=True)
def _msign_inner_compiled(X: torch.Tensor) -> torch.Tensor:
    """Compiled inner loop of msign - 8 Newton-Schulz iterations."""
    for i in range(8):
        a, b, c = MSIGN_COEFFS[i]
        X = _newton_schulz_step(X, a, b, c)
    return X


def msign_compiled(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Matrix sign via Newton-Schulz with torch.compile on the full iteration loop."""
    if G.ndim < 2:
        raise ValueError("Input tensor must have at least 2 dimensions.")
    if G.dtype != torch.float32:
        raise ValueError(f"Input tensor G must be in float32")

    transpose = G.size(-2) > G.size(-1)
    X = G.mT if transpose else G
    X = torch.nn.functional.normalize(X, p=2, dim=(-2, -1), eps=1e-7)
    
    X = _msign_inner_compiled(X)

    return X.mT if transpose else X


# ============================================================================
# Benchmark Functions
# ============================================================================

def benchmark_fn(fn, *args, warmup=3, repeat=10, **kwargs):
    """Benchmark a function and return mean time in milliseconds."""
    # Warmup
    for _ in range(warmup):
        result = fn(*args, **kwargs)
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)  # ms
    
    return min(times), sum(times) / len(times), max(times)


def main():
    print("=" * 100)
    print("Benchmark: PyTorch Native vs torch.compile vs Triton msign Implementation")
    print("=" * 100)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        sys.exit(1)
    
    device = torch.device("cuda:0")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print()
    
    # Test shapes
    SHAPES = [
        (128, 2048),
        (2048, 128),
        (2048, 2048),
        (2048, 4096),
        (4096, 2048),
        (2048, 6144),
        (6144, 2048),
    ]
    
    STEPS = 8
    
    # Import Triton implementation
    print("Importing Triton implementation...")
    from emerging_optimizers.orthogonalized_optimizers.spectral_ball_utils import _large_msign as msign_triton
    print("Done.")
    
    # Warmup torch.compile
    print("Warming up torch.compile...")
    warmup_tensor = torch.randn(128, 128, dtype=torch.float32, device=device)
    for _ in range(3):
        _ = msign_compiled(warmup_tensor, STEPS)
        torch.cuda.synchronize()
    print("Done.\n")
    
    # Store results
    results = []
    
    print("=" * 100)
    print(f"{'Shape':<15} | {'PyTorch (ms)':<15} | {'Compiled (ms)':<15} | {'Triton (ms)':<15} | {'Best':<10}")
    print("-" * 100)
    
    for shape in SHAPES:
        # Create input tensor
        G = torch.randn(*shape, dtype=torch.float32, device=device)
        
        # ---- PyTorch Native ----
        try:
            pt_min, pt_avg, pt_max = benchmark_fn(msign_pytorch, G.clone(), STEPS)
            pt_result = f"{pt_avg:.3f}"
        except Exception as e:
            pt_avg = float('inf')
            pt_result = f"ERR"
        
        # ---- torch.compile ----
        try:
            cp_min, cp_avg, cp_max = benchmark_fn(msign_compiled, G.clone(), STEPS)
            cp_result = f"{cp_avg:.3f}"
        except Exception as e:
            cp_avg = float('inf')
            cp_result = f"ERR"
        
        # ---- Triton ----
        try:
            tri_min, tri_avg, tri_max = benchmark_fn(msign_triton, G.clone(), STEPS)
            tri_result = f"{tri_avg:.3f}"
        except Exception as e:
            tri_avg = float('inf')
            tri_result = f"ERR"
        
        # Find best
        times = {'PyTorch': pt_avg, 'Compiled': cp_avg, 'Triton': tri_avg}
        best = min(times, key=times.get)
        best_time = times[best]
        
        print(f"{str(shape):<15} | {pt_result:<15} | {cp_result:<15} | {tri_result:<15} | {best:<10}")
        
        results.append({
            'shape': shape,
            'pytorch_ms': pt_avg,
            'compiled_ms': cp_avg,
            'triton_ms': tri_avg,
            'best': best
        })
        
        # Clean up
        del G
        torch.cuda.empty_cache()
    
    print("=" * 100)
    
    # Summary
    print("\nSummary by shape size:")
    small_threshold = 512
    
    small_shapes = [r for r in results if min(r['shape']) < small_threshold]
    large_shapes = [r for r in results if min(r['shape']) >= small_threshold]
    
    if small_shapes:
        print(f"\n  Small matrices (min dim < {small_threshold}):")
        for r in small_shapes:
            best = r['best']
            print(f"    {r['shape']}: Best = {best} ({min(r['pytorch_ms'], r['compiled_ms'], r['triton_ms']):.3f}ms)")
    
    if large_shapes:
        print(f"\n  Large matrices (min dim >= {small_threshold}):")
        for r in large_shapes:
            best = r['best']
            print(f"    {r['shape']}: Best = {best} ({min(r['pytorch_ms'], r['compiled_ms'], r['triton_ms']):.3f}ms)")
    
    print("\nRecommendation:")
    print("  - Small matrices: Use torch.compile or PyTorch native")
    print("  - Large matrices: Use Triton tsyrk kernel")
    print("\nNote:")
    print("  - PyTorch: torch.addmm + X @ X.T (FP32)")
    print("  - Compiled: torch.compile(mode='default') (FP32)")
    print("  - Triton: tsyrk kernel (BF16 compute)")


if __name__ == "__main__":
    main()

