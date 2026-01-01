"""Warmup script to pre-compile all Triton kernels before training.

Run this script once before training to compile and cache all kernels.
Subsequent runs will use the cached kernels.

Usage:
    python warmup_triton_kernels.py
"""
import os
import sys
import time
sys.path.insert(0, '/root/yangwang/Megatron-LM')

import torch

# Force medium precision for Triton kernels
torch.set_float32_matmul_precision("medium")

print("=" * 70)
print("Triton Kernel Warmup Script")
print("=" * 70)
print(f"Triton cache dir: ~/.triton/cache/")
print(f"Matmul precision: {torch.get_float32_matmul_precision()}")
print()

if not torch.cuda.is_available():
    print("ERROR: CUDA not available!")
    sys.exit(1)

device = torch.device("cuda:0")
print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
print()

# Import the modules (this may trigger some compilation)
print("Importing modules...")
start = time.time()
from emerging_optimizers.orthogonalized_optimizers.muon_utils import newton_schulz
from emerging_optimizers.orthogonalized_optimizers.spectral_ball_utils import msign
print(f"  Import time: {time.time() - start:.2f}s")
print()

# Define all the matrix sizes you use in training
# Add your actual sizes here!
WARMUP_SHAPES = [
    (128, 2048),
    (2048, 128),
    (2048, 2048),
    (2048, 4096),
    (4096, 2048),
    (2048, 6144),
    (6144, 2048),
    # Add more shapes as needed for your model
]

# msign coefficients
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

print("=" * 70)
print("Warming up newton_schulz kernels...")
print("=" * 70)

total_start = time.time()

for i, shape in enumerate(WARMUP_SHAPES):
    print(f"\n[{i+1}/{len(WARMUP_SHAPES)}] Shape: {shape}")
    sys.stdout.flush()
    
    # Create tensor
    G = torch.randn(*shape, dtype=torch.float32, device=device)
    
    # Warmup newton_schulz with custom coefficients (msign style)
    start = time.time()
    torch.cuda.synchronize()
    
    result = newton_schulz(
        x=G,
        steps=8,
        coefficient_type="custom",
        custom_coefficient_sets=MSIGN_COEFFS,
    )
    
    torch.cuda.synchronize()
    compile_time = time.time() - start
    
    # Second run to get actual runtime
    start = time.time()
    torch.cuda.synchronize()
    
    result = newton_schulz(
        x=G,
        steps=8,
        coefficient_type="custom",
        custom_coefficient_sets=MSIGN_COEFFS,
    )
    
    torch.cuda.synchronize()
    runtime = time.time() - start
    
    print(f"  Compile: {compile_time:.2f}s, Runtime: {runtime*1000:.2f}ms")
    sys.stdout.flush()
    
    # Clear GPU memory
    del G, result
    torch.cuda.empty_cache()

total_time = time.time() - total_start

print()
print("=" * 70)
print(f"Warmup completed in {total_time:.2f}s")
print("=" * 70)
print()
print("All kernels are now cached in ~/.triton/cache/")
print("Subsequent runs will skip compilation and use cached kernels.")
print()

