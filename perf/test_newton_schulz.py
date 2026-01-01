"""Debug script to find where newton_schulz hangs."""

import sys
sys.path.insert(0, '/root/yangwang/Megatron-LM')

print("Step 1: Importing torch...")
import torch
print("  Done.")

print("Step 2: Setting matmul precision to medium (FORCED)...")
torch.set_float32_matmul_precision("medium")
precision = torch.get_float32_matmul_precision()
print(f"  Precision set to: {precision}")

print("Step 3: Creating test tensor...")
torch.manual_seed(42)
x = torch.randn(64, 128, dtype=torch.float32)
print(f"  Tensor shape: {x.shape}, dtype: {x.dtype}")

print("Step 4: Manual newton_schulz implementation (bypass muon_utils)...")

# Inline implementation to debug
eps = 1e-7
coefficients = [(3.4445, -4.7750, 2.0315)]  # simple
steps = 5

print("  4a: Checking transpose condition...")
transpose = x.size(-2) > x.size(-1)
print(f"      transpose = {transpose}")

if transpose:
    x = x.mT
    print(f"      After transpose: {x.shape}")

print("  4b: Normalizing tensor...")
X = torch.nn.functional.normalize(x, p=2, dim=(-2, -1), eps=eps)
print(f"      Normalized tensor shape: {X.shape}")

print("  4c: Running Newton-Schulz iterations...")
for i in range(steps):
    a, b, c = coefficients[i % len(coefficients)]
    print(f"      Iteration {i+1}/{steps}: a={a}, b={b}, c={c}")
    
    print(f"        Computing A = X @ X.mT...")
    A = X @ X.mT
    print(f"        A shape: {A.shape}")
    
    print(f"        Computing B = torch.addmm(A, A, A, alpha=c, beta=b)...")
    B = torch.addmm(A, A, A, alpha=c, beta=b)
    print(f"        B shape: {B.shape}")
    
    print(f"        Computing X = torch.addmm(X, B, X, alpha=1.0, beta=a)...")
    X = torch.addmm(X, B, X, alpha=1.0, beta=a)
    print(f"        X shape: {X.shape}")

print("  4d: Undoing transpose if needed...")
if transpose:
    X = X.mT

print(f"  Final result shape: {X.shape}")

print("\n" + "=" * 60)
print("Now testing imports from muon_utils...")
print("=" * 60)

print("Step 5: Testing individual imports...")
sys.stdout.flush()

print("  5a: Importing typing...")
sys.stdout.flush()
from typing import Any, Literal
print("      Done.")
sys.stdout.flush()

print("  5b: Importing absl logging...")
sys.stdout.flush()
from absl import logging
print("      Done.")
sys.stdout.flush()

print("  5c: Importing triton_kernels (THIS MAY HANG)...")
sys.stdout.flush()
from emerging_optimizers import triton_kernels
print("      Done.")
sys.stdout.flush()

print("  5d: Importing newton_schulz...")
sys.stdout.flush()
from emerging_optimizers.orthogonalized_optimizers.muon_utils import newton_schulz
print("      Done.")
sys.stdout.flush()

print("Step 6: Skipping CPU test (use_syrk requires GPU)...")
sys.stdout.flush()

# # CPU test - skip since use_syrk is forced and requires GPU
# torch.manual_seed(42)
# x2 = torch.randn(64, 128, dtype=torch.float32)
# result = newton_schulz(x=x2, steps=5, coefficient_type="simple")
# print(f"  Result shape: {result.shape}")

print("\n" + "=" * 60)
print("Step 9: Testing msign with 2048x2048 on GPU...")
print("=" * 60)
sys.stdout.flush()

if torch.cuda.is_available():
    print("  Creating 2048x2048 tensor on GPU...")
    sys.stdout.flush()
    G = torch.randn(2048, 2048, dtype=torch.float32, device="cuda:0")
    print(f"  Tensor created: {G.shape}, {G.device}")
    sys.stdout.flush()
    
    # Test custom coefficients directly (same as msign)
    msign_coeffs = [
        (8.2051, -22.9019, 16.4607),
        (4.0664, -2.8612, 0.5184),
        (3.9096, -2.8234, 0.5250),
        (3.2856, -2.4153, 0.4853),
        (2.2779, -1.6198, 0.3985),
        (1.8726, -1.2307, 0.3585),
        (1.8564, -1.2132, 0.3568),
        (1.8750, -1.2500, 0.3750),
    ]
    
    # NOTE: muon_utils.py is modified to FORCE use newton_schulz_step_tsyrk (Triton syrk)
    # The use_syrk parameter is now ignored - always uses Triton kernel
    import time
    
    # Run 1: includes Triton JIT compilation time
    print("  Run 1: Calling newton_schulz (includes Triton JIT compile time)...")
    sys.stdout.flush()
    torch.cuda.synchronize()
    start = time.time()
    
    result_gpu = newton_schulz(
        x=G.clone(),
        steps=8,
        coefficient_type="custom",
        custom_coefficient_sets=msign_coeffs,
    )
    
    torch.cuda.synchronize()
    elapsed1 = time.time() - start
    print(f"  Result: {result_gpu.shape}, time: {elapsed1:.4f}s (includes JIT compile)")
    sys.stdout.flush()
    
    # Run 2: actual runtime (kernel already compiled)
    print("  Run 2: Calling newton_schulz (kernel already compiled)...")
    sys.stdout.flush()
    torch.cuda.synchronize()
    start = time.time()
    
    result_gpu2 = newton_schulz(
        x=G.clone(),
        steps=8,
        coefficient_type="custom",
        custom_coefficient_sets=msign_coeffs,
    )
    
    torch.cuda.synchronize()
    elapsed2 = time.time() - start
    print(f"  Result: {result_gpu2.shape}, time: {elapsed2:.4f}s (actual runtime)")
    sys.stdout.flush()
    
    # Run 3: one more time to confirm
    print("  Run 3: Calling newton_schulz (confirm)...")
    sys.stdout.flush()
    torch.cuda.synchronize()
    start = time.time()
    
    result_gpu3 = newton_schulz(
        x=G.clone(),
        steps=8,
        coefficient_type="custom",
        custom_coefficient_sets=msign_coeffs,
    )
    
    torch.cuda.synchronize()
    elapsed3 = time.time() - start
    print(f"  Result: {result_gpu3.shape}, time: {elapsed3:.4f}s")
    sys.stdout.flush()
    
    print(f"\n  Summary:")
    print(f"    Run 1 (with JIT): {elapsed1:.4f}s")
    print(f"    Run 2 (cached):   {elapsed2:.4f}s")
    print(f"    Run 3 (cached):   {elapsed3:.4f}s")
    print(f"    JIT overhead:     {elapsed1 - elapsed2:.4f}s")
    print("  ✓ Test PASSED!")
    sys.stdout.flush()
else:
    print("  CUDA not available, skipping GPU test")

print("\nAll steps completed successfully!")

