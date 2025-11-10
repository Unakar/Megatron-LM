# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Megatron spectral ball optimizer wrapper."""

import json
import logging
import os
from typing import Callable, List, Optional

import torch

from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import log_single_rank

from . import _get_param_groups, get_megatron_optimizer
from .layer_wise_optimizer import LayerWiseDistributedOptimizer
from .optimizer import (
    ChainedOptimizer,
    Float16OptimizerWithFloat16Params,
    FP32Optimizer,
    MegatronOptimizer,
)
from .optimizer_config import OptimizerConfig
from .spectral_ball import TensorParallelSpectralBall

logger = logging.getLogger(__name__)


# def _log_spectral_ball_params(
#     model_chunks: List[MegatronModule],
#     linear_params: List[torch.nn.Parameter],
#     nonlinear_params: List[torch.nn.Parameter],
#     output_json_path: Optional[str] = None,
#     verbose: bool = True,
# ):
#     """Log SpectralBall optimizer parameter information.

#     Args:
#         model_chunks: List of model chunks.
#         linear_params: List of linear parameters (optimized by SpectralBall).
#         nonlinear_params: List of nonlinear parameters (optimized by Adam).
#         output_json_path: Optional path to save JSON file.
#         verbose: Whether to print to console.
#     """
#     # Build param to name mapping
#     param_to_name = {}
#     for model_chunk in model_chunks:
#         for name, param in model_chunk.named_parameters():
#             param_to_name[id(param)] = name

#     # Collect linear param info
#     linear_param_info = []
#     for param in linear_params:
#         name = param_to_name.get(id(param), "unknown")
#         info = {
#             "name": name,
#             "shape": list(param.shape),
#             "numel": param.numel(),
#             "dtype": str(param.dtype),
#             "is_qkv": getattr(param, "is_qkv", False),
#             "expert_tp": getattr(param, "expert_tp", False),
#         }
#         linear_param_info.append(info)

#     # Collect nonlinear param info
#     nonlinear_param_info = []
#     for param in nonlinear_params:
#         name = param_to_name.get(id(param), "unknown")
#         info = {
#             "name": name,
#             "shape": list(param.shape),
#             "numel": param.numel(),
#             "dtype": str(param.dtype),
#         }
#         nonlinear_param_info.append(info)

#     # Summary statistics
#     total_linear_params = sum(p.numel() for p in linear_params)
#     total_nonlinear_params = sum(p.numel() for p in nonlinear_params)
#     total_params = total_linear_params + total_nonlinear_params

#     summary = {
#         "num_linear_params": len(linear_params),
#         "num_nonlinear_params": len(nonlinear_params),
#         "total_linear_numel": total_linear_params,
#         "total_nonlinear_numel": total_nonlinear_params,
#         "total_numel": total_params,
#         "linear_ratio": total_linear_params / total_params if total_params > 0 else 0,
#     }

#     result = {
#         "summary": summary,
#         "linear_params": linear_param_info,
#         "nonlinear_params": nonlinear_param_info,
#     }

#     # Print to console
#     rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
#     if verbose and rank == 0:
#         log_single_rank(logger, logging.INFO, "=" * 80)
#         log_single_rank(logger, logging.INFO, "SpectralBall Optimizer Parameter Summary")
#         log_single_rank(logger, logging.INFO, "=" * 80)
#         log_single_rank(
#             logger,
#             logging.INFO,
#             f"Total parameters: {summary['num_linear_params'] + summary['num_nonlinear_params']}",
#         )
#         log_single_rank(
#             logger, logging.INFO, f"  Linear (SpectralBall): {summary['num_linear_params']}"
#         )
#         log_single_rank(
#             logger, logging.INFO, f"  Nonlinear (Adam):      {summary['num_nonlinear_params']}"
#         )
#         log_single_rank(logger, logging.INFO, "")
#         log_single_rank(
#             logger,
#             logging.INFO,
#             f"Total elements: {summary['total_numel']:,} ({summary['total_numel']/1e6:.2f}M)",
#         )
#         log_single_rank(
#             logger,
#             logging.INFO,
#             f"  Linear:    {summary['total_linear_numel']:,} ({summary['linear_ratio']*100:.1f}%)",
#         )
#         log_single_rank(
#             logger,
#             logging.INFO,
#             f"  Nonlinear: {summary['total_nonlinear_numel']:,} ({(1-summary['linear_ratio'])*100:.1f}%)",
#         )
#         log_single_rank(logger, logging.INFO, "")
#         log_single_rank(logger, logging.INFO, "Linear Parameters (optimized by SpectralBall):")
#         log_single_rank(logger, logging.INFO, "-" * 80)

#         for i, info in enumerate(linear_param_info[:10]):  # Show first 10
#             qkv_flag = " [QKV]" if info["is_qkv"] else ""
#             expert_flag = " [Expert]" if info["expert_tp"] else ""
#             log_single_rank(
#                 logger,
#                 logging.INFO,
#                 f"  {i+1:3d}. {info['name']:60s} {str(info['shape']):20s}{qkv_flag}{expert_flag}",
#             )

#         if len(linear_param_info) > 10:
#             log_single_rank(
#                 logger, logging.INFO, f"  ... and {len(linear_param_info) - 10} more"
#             )
#         log_single_rank(logger, logging.INFO, "=" * 80)

#     # Save to JSON
#     if output_json_path and rank == 0:
#         dir_path = os.path.dirname(output_json_path)
#         if dir_path:
#             os.makedirs(dir_path, exist_ok=True)
#         with open(output_json_path, "w") as f:
#             json.dump(result, f, indent=2)
#         log_single_rank(logger, logging.INFO, f"Parameter info saved to: {output_json_path}")


def get_megatron_spectral_ball_optimizer(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    no_weight_decay_cond: Optional[Callable] = None,
    scale_lr_cond: Optional[Callable] = None,
    lr_mult: float = 1.0,
    use_gloo_process_groups: bool = True,
    layer_wise_distributed_optimizer: bool = False,
    pg_collection: Optional[ProcessGroupCollection] = None,
) -> MegatronOptimizer:
    """Get the spectral ball optimizer for model chunks.

    This function creates a chained optimizer where:
    - Linear weights (2D tensors) use SpectralBallOptimizer with spectral sphere constraints
    - Non-linear parameters (biases, norms, embeddings) use Adam

    Args:
        config: OptimizerConfig instance.
        model_chunks: List of model chunks to optimize.
        no_weight_decay_cond: Optional function to determine if a parameter should skip weight decay.
        scale_lr_cond: Optional function to determine if a parameter should use scaled learning rate.
        lr_mult: Learning rate multiplier for scaled parameters.
        use_gloo_process_groups: Whether to use Gloo process groups.
        layer_wise_distributed_optimizer: Whether to use layer-wise distributed optimization.
        pg_collection: Optional ProcessGroupCollection for distributed training.

    Returns:
        MegatronOptimizer instance (ChainedOptimizer or LayerWiseDistributedOptimizer).
    """
    # Distributed optimizer is not supported
    if config.use_distributed_optimizer:
        raise Exception('spectral_ball with distributed optimizer is not supported.')

    # Set up process groups
    if pg_collection is None:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        pg_collection.dp_cp = parallel_state.get_data_parallel_group(with_context_parallel=True)
        pg_collection.expt_dp = parallel_state.get_expert_data_parallel_group()

    log_single_rank(
        logger, logging.INFO, f'Setting up spectral ball optimizer with config {config}'
    )

    optimizers = []
    linear_params = []
    nonlinear_params = []

    # Categorize parameters into linear (2D) and non-linear (1D, embeddings)
    # Also tag QKV parameters and expert parameters
    for model_chunk in model_chunks:
        # Get QKV split shapes from model config
        num_attention_heads = model_chunk.config.num_attention_heads
        num_query_groups = model_chunk.config.num_query_groups
        kv_channels = model_chunk.config.kv_channels
        qkv_split_shapes = [
            num_attention_heads // num_query_groups * kv_channels,
            kv_channels,
            kv_channels,
        ]

        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue

            # Add flag for expert weight so optimizer can figure which tp group it uses
            if 'experts' in name and 'shared' not in name:
                param.expert_tp = True

            # Add flag for qkv parameter
            if 'linear_qkv.weight' in name and len(param.shape) == 2:
                param.is_qkv = True

            # Linear weights: 2D tensors that are not embeddings or output parameters
            if (
                not getattr(param, 'is_embedding_or_output_parameter', False)
                and len(param.shape) == 2
            ):
                linear_params.append(param)
            else:
                nonlinear_params.append(param)

    # # ==================== Log parameter information ====================
    # # Log parameter info and save to JSON
    # json_output_path = '/home/t2vg-a100-G2-1/a_xietian/dev/numeric/dev_logs/linear_matrix.json'
    # _log_spectral_ball_params(
    #     model_chunks=model_chunks,
    #     linear_params=linear_params,
    #     nonlinear_params=nonlinear_params,
    #     output_json_path=json_output_path,
    #     verbose=True,  # Always print to console
    # )

    # ==================== Setup SpectralBall for linear params ====================
    # Freeze non-linear params temporarily
    for param in nonlinear_params:
        param.requires_grad = False

    # Get param groups for linear params
    linear_param_groups = _get_param_groups(
        model_chunks,
        no_weight_decay_cond,
        scale_lr_cond,
        lr_mult,
        lr=config.lr,
        min_lr=config.min_lr,
        decoupled_lr=config.decoupled_lr,
        decoupled_min_lr=config.decoupled_min_lr,
    )

    # Create TensorParallelSpectralBall optimizer
    spectral_ball_optimizer = TensorParallelSpectralBall(
        linear_param_groups,
        lr=config.lr,
        momentum_beta=config.spectral_ball_momentum,
        use_nesterov=config.spectral_ball_use_nesterov,
        weight_decay=config.weight_decay,
        use_decoupled_weight_decay=config.decoupled_weight_decay,
        split_qkv=config.spectral_ball_split_qkv,
        is_qkv_fn=lambda p: getattr(p, 'is_qkv', False),
        qkv_split_shapes=tuple(qkv_split_shapes),
        msign_steps=config.spectral_ball_msign_steps,
        brent_tolerance_f=config.spectral_ball_brent_tol_f,
        brent_tolerance_x=config.spectral_ball_brent_tol_x,
        brent_max_iterations=config.spectral_ball_brent_max_iter,
        radius_mode=config.spectral_ball_radius_mode,
        pg_collection=pg_collection,
    )

    # Save original optimizer name and switch to adam for the rest
    original_optimizer = config.optimizer
    config.optimizer = 'adam'

    # Define init state function for SpectralBall
    def spectral_ball_init_state_fn(opt, config=None):
        """Initialize SpectralBall optimizer state for checkpointing."""
        for group in opt.param_groups:
            for p in group['params']:
                if len(opt.state[p]) == 0:
                    opt.state[p]['momentum_buffer'] = torch.zeros_like(p.data)
                    # Note: target_radius will be computed on first step

    # Define init state function for Adam
    def adam_init_state_fn(opt, config=None):
        """Initialize Adam optimizer state for checkpointing."""
        for group in opt.param_groups:
            for p in group['params']:
                if len(opt.state[p]) == 0:
                    if config is None or not config.use_precision_aware_optimizer:
                        opt.state[p]['exp_avg'] = torch.zeros_like(p.data)
                        opt.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)
                    else:
                        opt.initialize_state(p)

    # Wrap in precision-aware optimizer
    if config.fp16:
        raise Exception('spectral_ball with fp16 is not supported.')

    reset_config_bf16 = False
    if config.bf16:
        if layer_wise_distributed_optimizer:
            # Delay master weight creation for layer-wise sharding
            config.bf16 = False
            reset_config_bf16 = True
        else:
            spectral_ball_optimizer = Float16OptimizerWithFloat16Params(
                spectral_ball_optimizer, config, None, spectral_ball_init_state_fn
            )
    else:
        spectral_ball_optimizer = FP32Optimizer(
            spectral_ball_optimizer, config, spectral_ball_init_state_fn
        )

    optimizers.append(spectral_ball_optimizer)

    # ==================== Setup Adam for non-linear params ====================
    # Unfreeze non-linear params and freeze linear params
    for param in nonlinear_params:
        param.requires_grad = True
    for param in linear_params:
        param.requires_grad = False

    # Get Adam optimizer for non-linear params
    chained_adam = get_megatron_optimizer(
        config, model_chunks, no_weight_decay_cond, scale_lr_cond, lr_mult, use_gloo_process_groups
    )

    # Unfreeze all params
    for param in linear_params:
        param.requires_grad = True

    # Restore original optimizer name
    config.optimizer = original_optimizer

    # Chain optimizers together
    optimizers += chained_adam.chained_optimizers

    # ==================== Layer-wise distributed optimizer ====================
    if layer_wise_distributed_optimizer:
        log_single_rank(
            logger, logging.INFO, 'Using LayerWiseDistributedOptimizer for SpectralBall'
        )
        if reset_config_bf16:
            config.bf16 = True
        return LayerWiseDistributedOptimizer(
            optimizers,
            config,
            pg_collection,
            init_state_fn_list=[spectral_ball_init_state_fn, adam_init_state_fn],
        )

    return ChainedOptimizer(optimizers)
