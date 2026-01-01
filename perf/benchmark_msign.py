"""Benchmark script comparing PyTorch native vs Triton implementation of msign.

Compares:
1. PyTorch native: torch.addmm with X @ X.T
2. Triton: newton_schulz with use_syrk=True (tsyrk kernel)

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

def _pytorch_newton_schulz_step(X: torch.Tensor, a: float, b: float, c: float) -> torch.Tensor:
    """One Newton-Schulz iteration using PyTorch native ops: X ← a·X + X·(b·A + c·A²) where A = X·X^T."""
    A = X @ X.mT
    B = torch.addmm(A, A, A, alpha=c, beta=b)
    X = torch.addmm(X, B, X, alpha=1.0, beta=a)
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
        if i < 8:
            a, b, c = coeffs[i]
        else:
            a, b, c = coeffs[-1]
        X = _pytorch_newton_schulz_step(X, a, b, c)

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
    print("=" * 80)
    print("Benchmark: PyTorch Native vs Triton msign Implementation")
    print("=" * 80)
    
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
    from emerging_optimizers.orthogonalized_optimizers.spectral_ball_utils import msign as msign_triton
    print("Done.\n")
    
    # Store results
    results = []
    
    print("=" * 80)
    print(f"{'Shape':<15} | {'PyTorch (ms)':<20} | {'Triton (ms)':<20} | {'Speedup':<10}")
    print("-" * 80)
    
    for shape in SHAPES:
        # Create input tensor
        G = torch.randn(*shape, dtype=torch.float32, device=device)
        
        # ---- PyTorch Native ----
        try:
            pt_min, pt_avg, pt_max = benchmark_fn(msign_pytorch, G.clone(), STEPS)
            pt_result = f"{pt_avg:.3f} ± {(pt_max-pt_min)/2:.3f}"
        except Exception as e:
            pt_avg = float('inf')
            pt_result = f"ERROR: {e}"
        
        # ---- Triton (with medium precision for syrk) ----
        try:
            # msign_triton uses utils.fp32_matmul_precision("medium") context manager
            tri_min, tri_avg, tri_max = benchmark_fn(msign_triton, G.clone(), STEPS)
            tri_result = f"{tri_avg:.3f} ± {(tri_max-tri_min)/2:.3f}"
        except Exception as e:
            tri_avg = float('inf')
            tri_result = f"ERROR: {e}"
        
        # Calculate speedup
        if pt_avg != float('inf') and tri_avg != float('inf'):
            speedup = pt_avg / tri_avg
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup = 0
            speedup_str = "N/A"
        
        print(f"{str(shape):<15} | {pt_result:<20} | {tri_result:<20} | {speedup_str:<10}")
        
        results.append({
            'shape': shape,
            'pytorch_ms': pt_avg,
            'triton_ms': tri_avg,
            'speedup': speedup
        })
        
        # Clean up
        del G
        torch.cuda.empty_cache()
    
    print("=" * 80)
    
    # Summary
    print("\nSummary:")
    valid_speedups = [r['speedup'] for r in results if r['speedup'] > 0]
    if valid_speedups:
        print(f"  Average speedup: {sum(valid_speedups)/len(valid_speedups):.2f}x")
        print(f"  Min speedup: {min(valid_speedups):.2f}x")
        print(f"  Max speedup: {max(valid_speedups):.2f}x")
    
    print("\nNote:")
    print("  - PyTorch: Uses torch.addmm with X @ X.T (FP32)")
    print("  - Triton: Uses tsyrk kernel (BF16 compute, FP32 I/O)")
    print("  - First Triton run includes JIT compilation (cached for subsequent runs)")


if __name__ == "__main__":
    main()

