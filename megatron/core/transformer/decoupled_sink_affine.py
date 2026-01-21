# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""
Decoupled Sink Affine module for Residual Outlier Mitigation.

This module implements the "Dual Affine" architecture that decouples the responsibility
of creating outliers from dynamic activations to static learnable parameters.

Mathematical Formulation:
    x' = x ⊙ w_sink  (element-wise multiplication)
    output = LayerNorm(x')

Key Properties:
    1. w_sink is initialized to ones (1.0) - training starts identical to baseline
    2. w_sink must NOT have weight decay applied - expects to grow large
    3. Two independent instances per layer (one for Attention, one for MLP)
"""

import torch
import torch.nn as nn

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig


@torch.jit.script
def sink_affine_forward(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    JIT-compiled forward pass for better performance.
    
    Reduces Python launch overhead compared to plain x * weight.
    
    Args:
        x: Input tensor of shape [..., hidden_size]
        weight: Weight tensor of shape [hidden_size]
    
    Returns:
        Element-wise product x * weight
    """
    return x * weight


class DecoupledSinkAffine(MegatronModule):
    """
    Decoupled Sink Affine layer for residual stream outlier mitigation.
    
    This layer rescales the input before LayerNorm to transfer the responsibility
    of creating outliers from activations to learnable weights.
    
    The sink affine operation is: output = input * weight
    where weight is a learnable parameter of shape [hidden_size].
    
    Placement:
        Must be instantiated TWICE per transformer layer:
        1. One instance before input_layernorm (Self-Attention)
        2. One instance before pre_mlp_layernorm (MLP)
        These two instances have SEPARATE, INDEPENDENT learnable parameters.
    
    Sequence Parallelism Compatibility:
        - Input x may be sharded along sequence dimension when SP is enabled.
        - Element-wise multiplication with [hidden_size] weight is SP-safe.
        - No inter-GPU communication required for forward pass.
        - When SP is enabled, gradients are reduced across TP ranks via
          `sequence_parallel` attribute on the weight parameter.
    
    Weight Decay:
        The weight parameter is marked with `is_sink_affine = True` to ensure
        it is excluded from weight decay in the optimizer. This is critical
        because we expect the weight to grow large to absorb outliers.
    
    Performance Note:
        This layer introduces an additional HBM read/write operation and breaks
        the fusion of TE's FusedLayerNorm kernel. This may cause 1-3% throughput
        degradation. Use this feature when residual outlier mitigation is needed
        for quantization (FP8, W8A8) or training stability.
    
    Args:
        config: TransformerConfig object
        hidden_size: Size of the hidden dimension
    """
    
    def __init__(self, config: TransformerConfig, hidden_size: int):
        super().__init__(config=config)
        
        self.hidden_size = hidden_size
        
        # Initialize weight to ones - training starts identical to baseline
        self.weight = nn.Parameter(torch.ones(hidden_size))
        
        # Mark for no weight decay in optimizer
        # This is critical: we expect weight to grow large to absorb outliers
        self.weight.is_sink_affine = True
        
        # Set sequence_parallel attribute for proper gradient reduction in SP mode
        # When sequence_parallel is enabled, the input is sharded along the sequence
        # dimension across TP ranks. The weight gradient needs to be summed across
        # TP ranks to get the correct global gradient.
        # This aligns with how LayerNorm weights handle SP gradient reduction.
        setattr(self.weight, 'sequence_parallel', config.sequence_parallel)
        
        # Set allreduce=True to align with non-expert parameter behavior
        # This ensures the parameter participates in standard gradient AllReduce
        setattr(self.weight, 'allreduce', True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply sink affine transformation.
        
        Args:
            x: Input tensor of shape [seq_len, batch, hidden_size] or 
               [batch, seq_len, hidden_size] depending on Megatron config.
               May be sharded along sequence dimension when SP is enabled.
        
        Returns:
            Tensor of same shape as input, element-wise multiplied by weight.
        """
        # Ensure weight dtype matches input for mixed precision training
        # This handles BF16/FP16 activations with FP32 master weights
        weight = self.weight.type_as(x)
        
        return sink_affine_forward(x, weight)
    
    def extra_repr(self) -> str:
        return f'hidden_size={self.hidden_size}'
