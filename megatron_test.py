"""
Simple Megatron-LM environment test script.
Tests basic imports, GPU availability, and a minimal training loop.
"""
import os
import sys
import torch

# Add current directory to path
sys.path.insert(0, os.getcwd())


def check_env():
    """Check basic environment and dependencies."""
    print("=" * 60)
    print("ENVIRONMENT CHECK")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU 0: {torch.cuda.get_device_name(0)}")
    
    # Check optional dependencies
    try:
        import transformer_engine
        print(f"Transformer Engine: {transformer_engine.__version__}")
    except ImportError:
        print("Transformer Engine: NOT INSTALLED (optional for H100)")

    try:
        import apex
        print("Apex: INSTALLED")
    except ImportError:
        print("Apex: NOT INSTALLED (optional)")

    try:
        import megatron.core
        print("Megatron Core: IMPORTED SUCCESSFULLY")
    except ImportError as e:
        print(f"Megatron Core: IMPORT FAILED - {e}")
        return False
    
    return True


def run_simple_forward():
    """Run a simple forward pass with a tiny GPT model."""
    print("\n" + "=" * 60)
    print("SIMPLE FORWARD PASS TEST")
    print("=" * 60)
    
    from megatron.core import parallel_state
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    
    # Initialize distributed (single GPU)
    if not torch.distributed.is_initialized():
        os.environ.setdefault('RANK', '0')
        os.environ.setdefault('WORLD_SIZE', '1')
        os.environ.setdefault('MASTER_ADDR', 'localhost')
        os.environ.setdefault('MASTER_PORT', '29500')
        torch.cuda.set_device(0)
        torch.distributed.init_process_group(backend='nccl', rank=0, world_size=1)
    
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1
    )
    
    # Initialize the CUDA RNG tracker (this was missing!)
    model_parallel_cuda_manual_seed(42)
    
    print("Distributed initialized successfully")
    
    # Create a tiny model
    config = TransformerConfig(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
    )
    
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=100,
        max_sequence_length=64,
    ).cuda()
    
    print(f"Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create dummy input
    batch_size = 2
    seq_len = 64
    tokens = torch.randint(0, 100, (batch_size, seq_len), device='cuda')
    position_ids = torch.arange(seq_len, device='cuda').unsqueeze(0).expand(batch_size, -1)
    attention_mask = torch.ones(batch_size, 1, seq_len, seq_len, device='cuda', dtype=torch.bool)
    
    # Forward pass
    print("Running forward pass...")
    with torch.no_grad():
        output = model(tokens, position_ids, attention_mask)
    
    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    print(f"Output device: {output.device}")
    
    return True


def run_training_step():
    """Run a single training step with backward pass."""
    print("\n" + "=" * 60)
    print("TRAINING STEP TEST (Forward + Backward)")
    print("=" * 60)
    
    from megatron.core import parallel_state
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    
    # Ensure distributed is initialized
    if not torch.distributed.is_initialized():
        os.environ.setdefault('RANK', '0')
        os.environ.setdefault('WORLD_SIZE', '1')
        os.environ.setdefault('MASTER_ADDR', 'localhost')
        os.environ.setdefault('MASTER_PORT', '29500')
        torch.cuda.set_device(0)
        torch.distributed.init_process_group(backend='nccl', rank=0, world_size=1)
    
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1
        )
        # Initialize the CUDA RNG tracker
        model_parallel_cuda_manual_seed(42)
    
    # Create model
    config = TransformerConfig(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
    )
    
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=100,
        max_sequence_length=64,
    ).cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Create dummy input with labels
    batch_size = 2
    seq_len = 64
    tokens = torch.randint(0, 100, (batch_size, seq_len), device='cuda')
    position_ids = torch.arange(seq_len, device='cuda').unsqueeze(0).expand(batch_size, -1)
    attention_mask = torch.ones(batch_size, 1, seq_len, seq_len, device='cuda', dtype=torch.bool)
    labels = torch.randint(0, 100, (batch_size, seq_len), device='cuda')
    
    # Training step
    print("Running forward pass with labels...")
    optimizer.zero_grad()
    output = model(tokens, position_ids, attention_mask, labels=labels)
    
    # Compute loss (output is per-token loss when labels provided)
    loss = output.mean()
    print(f"Loss: {loss.item():.4f}")
    
    print("Running backward pass...")
    loss.backward()
    
    print("Running optimizer step...")
    optimizer.step()
    
    print("Training step completed successfully!")
    return True


def cleanup():
    """Clean up distributed resources."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MEGATRON-LM ENVIRONMENT TEST")
    print("=" * 60 + "\n")
    
    # Check environment
    if not check_env():
        print("\nFAILED: Basic environment check failed")
        sys.exit(1)
    
    if not torch.cuda.is_available():
        print("\nSKIPPED: No GPU available. Run this on a GPU node.")
        sys.exit(0)
    
    # Run tests
    try:
        run_simple_forward()
        run_training_step()
        
        print("\n" + "=" * 60)
        print("SUCCESS: All tests passed!")
        print("Your Megatron-LM environment is working correctly.")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        cleanup()
