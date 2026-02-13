"""Megatron HyperballAdam optimizer wrapper."""

import logging
from typing import Callable, List, Optional

import torch

from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import log_single_rank

from . import _get_param_groups, get_megatron_optimizer
from .optimizer import (
    ChainedOptimizer,
    Float16OptimizerWithFloat16Params,
    FP32Optimizer,
    MegatronOptimizer,
)
from .optimizer_config import OptimizerConfig
from emerging_optimizers.scalar_optimizers.hyperball_adam import HyperballAdam

logger = logging.getLogger(__name__)


def get_megatron_hyperball_adam_optimizer(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    no_weight_decay_cond: Optional[Callable] = None,
    scale_lr_cond: Optional[Callable] = None,
    lr_mult: float = 1.0,
    use_gloo_process_groups: bool = True,
) -> MegatronOptimizer:
    """Get the HyperballAdam optimizer for model chunks.

    This function creates a chained optimizer where:
    - Linear weights (2D tensors) use HyperballAdam with Frobenius-norm sphere constraint
    - Non-linear parameters (biases, norms, embeddings) use standard Adam

    The update rule for 2D weights:
        W_{t+1} = R * Normalize(W_t - lr * R * Normalize(u_t))
    where R = ||W_0||_F and u_t is the Adam update direction.

    Args:
        config: OptimizerConfig instance.
        model_chunks: List of model chunks to optimize.
        no_weight_decay_cond: Optional function to determine if a parameter should skip weight decay.
        scale_lr_cond: Optional function to determine if a parameter should use scaled learning rate.
        lr_mult: Learning rate multiplier for scaled parameters.
        use_gloo_process_groups: Whether to use Gloo process groups.

    Returns:
        MegatronOptimizer instance (ChainedOptimizer).
    """
    # Distributed optimizer is not supported
    if config.use_distributed_optimizer:
        raise Exception('hyperball_adam with distributed optimizer is not supported.')

    log_single_rank(
        logger, logging.INFO, f'Setting up HyperballAdam optimizer with config {config}'
    )

    optimizers = []
    linear_params = []
    nonlinear_params = []

    # Categorize parameters into linear (2D) and non-linear (1D, embeddings)
    for model_chunk in model_chunks:
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue

            # Store parameter name for logging
            param.param_name = name

            # Linear weights: 2D tensors that are not embeddings or output parameters
            if (
                not getattr(param, 'is_embedding_or_output_parameter', False)
                and len(param.shape) == 2
            ):
                linear_params.append(param)
            else:
                nonlinear_params.append(param)

    # ==================== Setup HyperballAdam for linear params ====================
    # Freeze non-linear params temporarily
    for param in nonlinear_params:
        param.requires_grad = False

    # Get param groups for linear params
    # Force all linear params to have wd_mult=0.0 (no weight decay for linear layers)
    # HyperballAdam constrains weights to the Frobenius sphere, so weight decay is unnecessary
    linear_no_weight_decay_cond = lambda name, param: True  # All linear params skip weight decay
    linear_param_groups = _get_param_groups(
        model_chunks,
        linear_no_weight_decay_cond,
        scale_lr_cond,
        lr_mult,
        lr=config.lr,
        min_lr=config.min_lr,
        decoupled_lr=config.decoupled_lr,
        decoupled_min_lr=config.decoupled_min_lr,
    )

    # Create HyperballAdam optimizer
    hyperball_adam_optimizer = HyperballAdam(
        linear_param_groups,
        lr=config.lr,
        betas=(config.hyperball_adam_beta1, config.hyperball_adam_beta2),
        eps=config.hyperball_adam_eps,
        weight_decay=0.0,  # No weight decay needed — sphere constraint replaces it
        bias_correction=config.hyperball_adam_bias_correction,
    )

    # Save original optimizer name and switch to adam for the rest
    original_optimizer = config.optimizer
    config.optimizer = 'adam'

    # Define init state function for HyperballAdam
    def hyperball_adam_init_state_fn(opt, config=None):
        """Initialize HyperballAdam optimizer state for checkpointing."""
        for group in opt.param_groups:
            for p in group['params']:
                if len(opt.state[p]) == 0:
                    opt.state[p]['exp_avg'] = torch.zeros_like(p.data)
                    opt.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)
                    opt.state[p]['step'] = 0
                    opt.state[p]['initial_frobenius_norm'] = torch.norm(
                        p.data.float(), p='fro'
                    ).item()

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
        raise Exception('hyperball_adam with fp16 is not supported.')

    if config.bf16:
        hyperball_adam_optimizer = Float16OptimizerWithFloat16Params(
            hyperball_adam_optimizer, config, None, hyperball_adam_init_state_fn
        )
    else:
        hyperball_adam_optimizer = FP32Optimizer(
            hyperball_adam_optimizer, config, hyperball_adam_init_state_fn
        )

    optimizers.append(hyperball_adam_optimizer)

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

    return ChainedOptimizer(optimizers)
