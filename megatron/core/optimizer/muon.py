# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Megatron muon optimizer wrapper to handle tensor-parallel."""

import logging
from typing import Any, Callable, List, Literal, Optional

import torch
from torch.optim.optimizer import ParamsT

from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import get_pg_size, log_single_rank

from . import _get_param_groups, get_megatron_optimizer
from .layer_wise_optimizer import LayerWiseDistributedOptimizer
from .optimizer import (
    ChainedOptimizer,
    Float16OptimizerWithFloat16Params,
    FP32Optimizer,
    MegatronOptimizer,
)
from .optimizer_config import OptimizerConfig

try:
    from emerging_optimizers.orthogonalized_optimizers import (
        OrthogonalizedOptimizer,
        get_muon_scale_factor,
    )
    from emerging_optimizers.orthogonalized_optimizers.muon_utils import newton_schulz_tp
    from emerging_optimizers import mixin as opt_mixin

    HAVE_EMERGING_OPTIMIZERS = True
except ImportError:
    HAVE_EMERGING_OPTIMIZERS = False
    OrthogonalizedOptimizer = object


logger = logging.getLogger(__name__)


class TensorParallelMuon(OrthogonalizedOptimizer):
    """Tensor Parallel Muon optimizer."""

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum_beta: float = 0.95,
        use_nesterov: bool = True,
        weight_decay: float = 0.01,
        use_decoupled_weight_decay: bool = True,
        split_qkv: bool = False,
        is_qkv_fn: Callable[[torch.Tensor], bool] | None = None,
        qkv_split_shapes: tuple[int, int, int] | None = None,
        qkv_split_mode: str = "component",  # "component", "group", or "head"
        # FC1 split support for gated linear units (SwiGLU)
        split_fc1: bool = False,
        is_fc1_fn: Callable[[torch.Tensor], bool] | None = None,
        fc1_split_shapes: tuple[int, int] | None = None,  # (gate_dim, up_dim)
        # MoE expert split support for GroupedMLP
        split_moe_experts: bool = False,
        is_grouped_moe_fn: Callable[[torch.Tensor], bool] | None = None,
        fp32_matmul_prec: str = "medium",
        coefficient_type: str = "polar_express",
        num_ns_steps: int = 5,
        scale_mode: str = "spectral",
        extra_scale_factor: float = 1.0,
        pg_collection: Optional[ProcessGroupCollection] = None,
        mode: Literal["blockwise", "duplicated", "distributed"] = "duplicated",
    ) -> None:
        if num_ns_steps < 1:
            raise ValueError(f"num_ns_steps must be at least 1, got {num_ns_steps}")

        def scaled_orthogonalize_fn(
            grad: torch.Tensor,
            tp_group: torch.distributed.ProcessGroup,
            partition_dim: int | None = None,
        ) -> torch.Tensor:
            log_single_rank(
                logger,
                logging.DEBUG,
                f'Orthogonalizing grad with {num_ns_steps} steps, {coefficient_type} coefficient, '
                f'{scale_mode} scale mode, extra_scale_factor={extra_scale_factor}',
            )
            size = [grad.size(-2), grad.size(-1)]
            if partition_dim:
                size[partition_dim] *= get_pg_size(tp_group)
            orth_grad = newton_schulz_tp(
                grad,
                steps=num_ns_steps,
                coefficient_type=coefficient_type,
                tp_group=tp_group,
                partition_dim=partition_dim,
                mode="duplicated" if mode == "blockwise" else mode,
            )
            scale_factor = get_muon_scale_factor(size[0], size[1], mode=scale_mode)
            return orth_grad * scale_factor * extra_scale_factor

        self.pg_collection = pg_collection
        self.mode = mode
        self.split_qkv = split_qkv
        self.is_qkv_fn = is_qkv_fn
        self.qkv_split_shapes = qkv_split_shapes
        self.qkv_split_mode = qkv_split_mode
        # FC1 split for gated linear units
        self.split_fc1 = split_fc1
        self.is_fc1_fn = is_fc1_fn
        self.fc1_split_shapes = fc1_split_shapes
        # MoE expert split for GroupedMLP
        self.split_moe_experts = split_moe_experts
        self.is_grouped_moe_fn = is_grouped_moe_fn

        # https://github.com/NVIDIA-NeMo/Emerging-Optimizers/blob/fe29e5670fc0dadf1f10ab267a0edfa6e1b89fb3/emerging_optimizers/orthogonalized_optimizers/muon.py#L71
        if use_decoupled_weight_decay:
            self.weight_decay_method = "decoupled"
        else:
            raise NotImplementedError

        super().__init__(
            params,
            lr,
            momentum_beta,
            weight_decay,
            use_nesterov=use_nesterov,
            weight_decay_method=self.weight_decay_method,
            fp32_matmul_prec=fp32_matmul_prec,
            scaled_orthogonalize_fn=scaled_orthogonalize_fn,
            log_per_module_update_rms=False,  # Will be set later via config
        )

    def orthogonalize(self, p: torch.Tensor, grad: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Orthogonalize the momentum.

        Args:
            p: The parameter tensor. i is necessary to pass param tensor in addition to momentum
                because a lot of information is only available in the param tensor,
                attributes for example.
            grad: The momentum tensor.

        Returns:
            The orthogonalized gradient tensor.
        """
        # TODO(deyuf): switch to group
        if self.pg_collection:
            tp_group = (
                self.pg_collection.expt_tp
                if getattr(p, 'expert_tp', False)
                else self.pg_collection.tp
            )
        else:
            tp_group = None
        partition_dim = None if self.mode == "blockwise" else getattr(p, "partition_dim", None)
        if partition_dim == -1:
            # llm-shower use different default value for partition_dim than TE.
            # Because -1 is a valid index for ndarray, we decided to not overload it.
            partition_dim = None

        if self.split_moe_experts and self.is_grouped_moe_fn is not None and self.is_grouped_moe_fn(p):  # type: ignore[misc]
            # Split GroupedMLP weight1/weight2 by experts
            # weight1: [hidden_size, num_experts * ffn_hidden_size] (or partitioned version)
            # weight2: [num_experts * ffn_hidden_size, hidden_size] (or partitioned version)
            num_local_experts = getattr(p, 'num_local_experts', None)
            if num_local_experts is None or num_local_experts <= 1:
                # If num_local_experts is not set or is 1, fall back to default behavior
                grad = self.scaled_orthogonalize_fn(grad, tp_group, partition_dim)
            else:
                grad_shape = grad.shape
                param_name = getattr(p, 'param_name', '')

                log_single_rank(
                    logger,
                    logging.DEBUG,
                    f'MoE expert split for {param_name}, grad shape {grad_shape}, num_experts {num_local_experts}',
                )

                if 'weight1' in param_name:
                    # weight1: [hidden_size, num_experts * ffn_per_expert]
                    # Need to account for gated linear units which double the output size
                    is_gated = getattr(p, 'is_gated', False)
                    ffn_multiplier = 2 if is_gated else 1
                    ffn_dim_per_expert = grad_shape[1] // (num_local_experts * ffn_multiplier)

                    # Reshape to separate experts: [hidden_size, num_experts, ffn_per_expert * multiplier]
                    grad_reshaped = grad.view(grad_shape[0], num_local_experts, ffn_dim_per_expert * ffn_multiplier)

                    # Orthogonalize each expert independently
                    expert_grads = []
                    for expert_idx in range(num_local_experts):
                        expert_grad = grad_reshaped[:, expert_idx, :]  # [hidden_size, ffn_per_expert * multiplier]

                        # Further split gate and up if split_fc1 is enabled and this is a gated layer
                        if self.split_fc1 and is_gated:
                            # Split into gate and up: each is [hidden_size, ffn_per_expert]
                            gate_grad, up_grad = torch.split(expert_grad, [ffn_dim_per_expert, ffn_dim_per_expert], dim=1)

                            # Orthogonalize gate and up independently
                            gate_grad_orth = self.scaled_orthogonalize_fn(gate_grad, tp_group, partition_dim)
                            up_grad_orth = self.scaled_orthogonalize_fn(up_grad, tp_group, partition_dim)

                            # Concatenate gate and up back together
                            expert_grad_orth = torch.cat([gate_grad_orth, up_grad_orth], dim=1)
                        else:
                            # Process the entire expert gradient as a single matrix
                            expert_grad_orth = self.scaled_orthogonalize_fn(expert_grad, tp_group, partition_dim)

                        expert_grads.append(expert_grad_orth)

                    # Merge back to original shape
                    grad = torch.stack(expert_grads, dim=1).view(grad_shape)

                elif 'weight2' in param_name:
                    # weight2: [num_experts * ffn_hidden_size_per_partition, hidden_size]
                    # Note: in TP case, the first dimension is already partitioned
                    ffn_dim_per_expert = grad_shape[0] // num_local_experts

                    # Reshape to separate experts: [num_experts, ffn_per_expert, hidden_size]
                    grad_reshaped = grad.view(num_local_experts, ffn_dim_per_expert, grad_shape[1])

                    # Orthogonalize each expert independently
                    expert_grads = []
                    for expert_idx in range(num_local_experts):
                        expert_grad = grad_reshaped[expert_idx, :, :]  # [ffn_per_expert, hidden_size]
                        expert_grad_orth = self.scaled_orthogonalize_fn(expert_grad, tp_group, partition_dim)
                        expert_grads.append(expert_grad_orth)

                    # Merge back to original shape
                    grad = torch.stack(expert_grads, dim=0).view(grad_shape)
                else:
                    # Unknown parameter name, fall back to default
                    grad = self.scaled_orthogonalize_fn(grad, tp_group, partition_dim)
        elif self.split_qkv and self.is_qkv_fn(p):  # type: ignore[misc]
            # split grouped attention parameters (e.g., QKV, GQA, etc.)
            assert self.qkv_split_shapes is not None, "qkv_split_shapes must be provided for head mode"
            grad_shape = grad.shape
            log_single_rank(
                logger,
                logging.DEBUG,
                f'qkv split grad shape {grad_shape}, split shapes {self.qkv_split_shapes}',
            )
            num_query_groups = grad_shape[0] // sum(self.qkv_split_shapes)
            grad_view = grad.view(num_query_groups, sum(self.qkv_split_shapes), -1)

            q_dim_per_group, kv_channels, _ = self.qkv_split_shapes  # v_dim not used (same as k_dim/kv_channels)
            heads_per_group = q_dim_per_group // kv_channels

            if self.qkv_split_mode == "group":
                # Group mode: process each query group independently, with Q/K/V split within each group
                # This aligns with split_qkv_init which initializes each query group independently
                group_updates = []
                for g in range(num_query_groups):
                    # Split this group into Q/K/V
                    qkv_comps = torch.split(grad_view[g], list(self.qkv_split_shapes), dim=0)
                    # Apply Newton-Schulz to each component
                    comp_updates = [
                        self.scaled_orthogonalize_fn(comp, tp_group, partition_dim)
                        for comp in qkv_comps
                    ]
                    # Concatenate Q/K/V updates within this group
                    group_updates.append(torch.cat(comp_updates, dim=0))
                # Stack all groups and reshape
                grad = torch.stack(group_updates, dim=0).view(grad_shape)
            elif self.qkv_split_mode == "head":
                # Head mode: process each attention head independently for Q/K/V
                group_updates = []
                for g in range(num_query_groups):
                    # Split this group into Q/K/V
                    qkv_comps = torch.split(grad_view[g], list(self.qkv_split_shapes), dim=0)
                    q_grad, k_grad, v_grad = qkv_comps

                    # Q: split into individual heads and process each
                    # q_grad shape: [heads_per_group * kv_channels, hidden_dim]
                    q_grad_heads = q_grad.view(heads_per_group, kv_channels, -1)

                    q_head_updates = []
                    for h in range(heads_per_group):
                        uh = self.scaled_orthogonalize_fn(q_grad_heads[h], tp_group, partition_dim)
                        q_head_updates.append(uh)

                    # Merge Q head updates
                    q_grad_updated = torch.stack(q_head_updates, dim=0).reshape(q_dim_per_group, -1)

                    # K and V: process directly
                    k_grad_updated = self.scaled_orthogonalize_fn(k_grad, tp_group, partition_dim)
                    v_grad_updated = self.scaled_orthogonalize_fn(v_grad, tp_group, partition_dim)

                    # Concatenate Q/K/V updates within this group
                    group_updates.append(torch.cat([q_grad_updated, k_grad_updated, v_grad_updated], dim=0))

                # Stack all groups and reshape
                grad = torch.stack(group_updates, dim=0).view(grad_shape)
            else:  # component mode (original logic)
                # Component mode: merge all groups' Q together, all K together, all V together
                qkv_grads = torch.split(grad_view, list(self.qkv_split_shapes), dim=1)
                qkv_grads = [g.reshape(-1, grad_shape[-1]) for g in qkv_grads]

                # Apply Newton-Schulz and scales to each component, concat back
                qkv_grads = [
                    self.scaled_orthogonalize_fn(g, tp_group, partition_dim).view(
                        num_query_groups, -1, grad_shape[-1]
                    )
                    for g in qkv_grads
                ]
                grad = torch.cat(qkv_grads, dim=1).view(grad_shape)
        elif self.split_fc1 and self.is_fc1_fn is not None and self.is_fc1_fn(p):
            # Split FC1 (gate and up) for gated linear units (SwiGLU)
            assert self.fc1_split_shapes is not None, "fc1_split_shapes must be provided for fc1 split"
            grad_shape = grad.shape
            gate_dim, up_dim = self.fc1_split_shapes
            log_single_rank(
                logger,
                logging.DEBUG,
                f'fc1 split grad shape {grad_shape}, split shapes {self.fc1_split_shapes}',
            )

            # Split gate and up along dim=0
            gate_grad, up_grad = torch.split(grad, [gate_dim, up_dim], dim=0)

            # Apply Newton-Schulz to each component
            gate_grad = self.scaled_orthogonalize_fn(gate_grad, tp_group, partition_dim)
            up_grad = self.scaled_orthogonalize_fn(up_grad, tp_group, partition_dim)

            # Concatenate back
            grad = torch.cat([gate_grad, up_grad], dim=0)
        else:
            grad = self.scaled_orthogonalize_fn(grad, tp_group, partition_dim)
        return grad


def get_megatron_muon_optimizer(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    no_weight_decay_cond: Optional[Callable] = None,
    scale_lr_cond: Optional[Callable] = None,
    lr_mult: float = 1.0,
    use_gloo_process_groups: bool = True,
    layer_wise_distributed_optimizer: bool = False,
    pg_collection: Optional[ProcessGroupCollection] = None,
) -> MegatronOptimizer:
    """This function is used to get the muon optimizer for the model chunks.
    It is used to get the muon optimizer for the model chunks.

    Args:
        config (OptimizerConfig): optimizer configuration object.
        model_chunks (List[MegatronModule]): model chunks to get optimizer for.
        no_weight_decay_cond (func, optional): function to determine whether a parameter
            should not perform weight decay. Defaults to None.
        scale_lr_cond (func, optional): function to determine whether a parameter
            should have a scaled learning rate. Defaults to None.
        lr_mult (float, optional): learning rate multiplier for parameters that
            satisfy scale_lr_cond. Defaults to 1.0.
        use_gloo_process_groups (bool): if false, disable use of Gloo process groups
            in underlying Megatron optimizers.
        layer_wise_distributed_optimizer (bool): if true, use layer-wise distributed optimizer.
            Defaults to False.
    """
    assert HAVE_EMERGING_OPTIMIZERS, "Emerging Optimizers is not installed."

    # dist-optim is not supported due to strong coupling with how DDP init grad buffer
    # in thoery we can put some weight to use non-dist-muon and rest to dist-adam
    # but there are strong dependency and assumption in DDP that prevent it
    if config.use_distributed_optimizer:
        raise Exception('muon with dist optimizer is not supported.')

    # before this function receive properly created collection
    if pg_collection is None:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        pg_collection.dp_cp = parallel_state.get_data_parallel_group(with_context_parallel=True)
        pg_collection.expt_dp = parallel_state.get_expert_data_parallel_group()

    log_single_rank(logger, logging.INFO, f'Setting up emerging optimizer with config {config}')

    optimizers = []
    # record list of non/linear params
    linear_params = []
    nonlinear_params = []

    qkv_split_shapes = None
    fc1_split_shapes = None
    for model_chunk in model_chunks:
        # use config to determine qkv split shapes.
        # no need to check tp since tp splits by head and this is per head(group) dimension
        num_attention_heads = model_chunk.config.num_attention_heads
        num_query_groups = model_chunk.config.num_query_groups
        kv_channels = model_chunk.config.kv_channels
        qkv_split_shapes = [
            num_attention_heads // num_query_groups * kv_channels,
            kv_channels,
            kv_channels,
        ]
        # derive fc1 split shapes for gated linear units (SwiGLU)
        try:
            if model_chunk.config.gated_linear_unit:
                ffn_hidden_size = model_chunk.config.ffn_hidden_size
                fc1_split_shapes = [ffn_hidden_size, ffn_hidden_size]  # gate, up
        except Exception:
            pass
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue
            # Store parameter name for logging
            param.param_name = name
            # add flag for expert weight so optimizer can figure which tp group it uses
            # alternatively, create new param group and save tp_group. this require more
            # change in optimizer
            if 'experts' in name and 'shared' not in name:
                param.expert_tp = True
            # add flag for qkv parameter
            # TODO(deyuf): support MLA
            if 'linear_qkv.weight' in name and len(param.shape) == 2:
                param.is_qkv = True
            # add flag for fc1 parameter (gated linear units like SwiGLU)
            if 'linear_fc1.weight' in name and len(param.shape) == 2:
                param.is_fc1 = True
            # add flag for GroupedMLP weight1/weight2 (MoE experts)
            if 'experts.weight1' in name or 'experts.weight2' in name:
                param.is_grouped_moe = True
                # Store MoE configuration for expert splitting
                try:
                    param.num_local_experts = model_chunk.config.num_moe_experts // model_chunk.config.expert_model_parallel_size
                    param.moe_ffn_hidden_size = model_chunk.config.moe_ffn_hidden_size
                    param.is_gated = model_chunk.config.gated_linear_unit
                except Exception:
                    # If config not available, disable expert splitting for this param
                    param.is_grouped_moe = False
            # TODO(deyuf): might not be sufficient for future algorithm. revisit this conditioning
            if not getattr(param, 'is_embedding_or_output_parameter', False) and not (
                len(param.shape) == 1
            ):
                linear_params.append(param)
            else:
                nonlinear_params.append(param)

    # freezing nonlinear params and get param groups for muon
    for param in nonlinear_params:
        param.requires_grad = False

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

    optimizer = TensorParallelMuon(
        linear_param_groups,
        lr=config.lr,
        momentum_beta=config.muon_momentum,
        use_nesterov=config.muon_use_nesterov,
        weight_decay=config.weight_decay,
        fp32_matmul_prec=config.muon_fp32_matmul_prec,
        num_ns_steps=config.muon_num_ns_steps,
        scale_mode=config.muon_scale_mode,
        split_qkv=config.muon_split_qkv,
        is_qkv_fn=lambda p: getattr(p, 'is_qkv', False),
        qkv_split_shapes=qkv_split_shapes,
        qkv_split_mode=config.muon_qkv_split_mode,
        split_fc1=config.muon_split_fc1,
        is_fc1_fn=lambda p: getattr(p, 'is_fc1', False),
        fc1_split_shapes=tuple(fc1_split_shapes) if fc1_split_shapes is not None else None,
        split_moe_experts=config.muon_split_moe_experts,
        is_grouped_moe_fn=lambda p: getattr(p, 'is_grouped_moe', False),
        extra_scale_factor=config.muon_extra_scale_factor,
        pg_collection=pg_collection,
        mode=config.muon_tp_mode,
    )

    # Enable per-module logging if configured
    optimizer.log_per_module_update_rms = config.log_per_module_update_rms
    if hasattr(optimizer, 'log_per_module_grad_rms'):
        optimizer.log_per_module_grad_rms = config.log_per_module_grad_rms

    # set config here to:
    # 1. get adam for rest of layer
    # 2. avoid ChainedOptimizer check fail that assert all optimizers are same kind
    # side effect is muon optimizer will have wrong name str, i.e. config.optimizer == 'adam'
    # TODO(deyuf): allow user to select optimizer mix and relax ChainedOptimizer design
    config.optimizer = 'adam'

    # Needed for torch_dist ckpt_format, unlike torch ckpt_format
    # For other emerging optimizers, need to implement init_state_fn as well
    # TODO(boxiangw): Improve usability after optimizer refactor
    # TODO(boxiangw): support precision aware optimizer
    def muon_init_state_fn(opt, config=None):
        for group in opt.param_groups:
            for p in group['params']:
                if len(opt.state[p]) == 0:
                    opt.state[p]['momentum_buffer'] = torch.zeros_like(p.data)

    def adam_init_state_fn(opt, config=None):
        for group in opt.param_groups:
            for p in group['params']:
                if len(opt.state[p]) == 0:
                    if config is None or not config.use_precision_aware_optimizer:
                        opt.state[p]['exp_avg'] = torch.zeros_like(p.data)
                        opt.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)
                    else:
                        opt.initialize_state(p)

    # need to wrap into megatron mix precision optimizer. (only support bf16 w/o loss scale now)
    if config.fp16:
        raise Exception('muon with fp16 is not supported.')

    reset_config_bf16 = False
    if config.bf16:
        if layer_wise_distributed_optimizer:
            # creating master weight before layerwise sharding will lead to unnecessary master
            # weight so here we delay master weight creation into layer_wise unset config.bf16
            # will also result in all optimizers below(adam) to also not be wrapped
            config.bf16 = False
            reset_config_bf16 = True
        else:
            # if not using layer_wise wrapper, just create master weight here is fine
            optimizer = Float16OptimizerWithFloat16Params(
                optimizer, config, None, muon_init_state_fn
            )
    else:
        optimizer = FP32Optimizer(optimizer, config, muon_init_state_fn)

    optimizers.append(optimizer)

    # done with muon, unfreeze nonlinear and freeze linear
    for param in nonlinear_params:
        param.requires_grad = True
    for param in linear_params:
        param.requires_grad = False

    # call original get. linear params will be skipped since they're freezed
    chained_adam = get_megatron_optimizer(
        config, model_chunks, no_weight_decay_cond, scale_lr_cond, lr_mult, use_gloo_process_groups
    )

    # unfreeze everything
    for param in linear_params:
        param.requires_grad = True

    # chain everything together
    optimizers += chained_adam.chained_optimizers

    if layer_wise_distributed_optimizer:
        log_single_rank(logger, logging.INFO, 'Using LayerWiseDistributedOptimizer for Muon')
        if reset_config_bf16:
            config.bf16 = True
        return LayerWiseDistributedOptimizer(
            optimizers,
            config,
            pg_collection,
            init_state_fn_list=[muon_init_state_fn, adam_init_state_fn],
        )
    return ChainedOptimizer(optimizers)
