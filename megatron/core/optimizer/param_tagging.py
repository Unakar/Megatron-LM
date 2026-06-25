import logging
from dataclasses import dataclass
from typing import List, Optional

import torch

from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import log_single_rank


@dataclass
class ParamBucketResult:
    linear_params: List[torch.Tensor]
    nonlinear_params: List[torch.Tensor]
    qkv_split_shapes: Optional[list[int]]
    fc1_split_shapes: Optional[list[int]]


def tag_and_bucket_params(
    model_chunks: List[MegatronModule],
    *,
    optimizer_name: str,
    include_router_flag: bool,
    include_muon_extended_flags: bool,
    use_attention_output_gate_for_qkv: bool,
    muon_vectorize: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None,
) -> ParamBucketResult:
    """Tag trainable parameters and split them into linear/nonlinear buckets.

    The tagging logic is shared across Muon, SpectralBall and MuonBall, while
    each optimizer controls behavior with feature flags.
    """
    linear_params: List[torch.Tensor] = []
    nonlinear_params: List[torch.Tensor] = []
    qkv_split_shapes: Optional[list[int]] = None
    fc1_split_shapes: Optional[list[int]] = None

    tag_counts = {
        "expert_tp": 0,
        "router": 0,
        "qkv": 0,
        "fc1": 0,
        "fc2": 0,
        "o_proj": 0,
        "embedding": 0,
        "lm_head": 0,
        "grouped_moe": 0,
        "moe_fc1": 0,
        "moe_fc2": 0,
    }
    trainable_params = 0

    muon_vectorize = muon_vectorize or []

    for model_chunk in model_chunks:
        # Derive qkv split shapes from model config when available.
        try:
            num_attention_heads = model_chunk.config.num_attention_heads
            num_query_groups = model_chunk.config.num_query_groups
            kv_channels = model_chunk.config.kv_channels
            if use_attention_output_gate_for_qkv and getattr(model_chunk.config, "attention_output_gate", False):
                qkv_split_shapes = [
                    num_attention_heads // num_query_groups * kv_channels,
                    num_attention_heads // num_query_groups * kv_channels,
                    kv_channels,
                    kv_channels,
                ]
            else:
                qkv_split_shapes = [
                    num_attention_heads // num_query_groups * kv_channels,
                    kv_channels,
                    kv_channels,
                ]
        except Exception:
            pass

        # Derive fc1 split shapes for SwiGLU.
        try:
            if model_chunk.config.gated_linear_unit:
                ffn_hidden_size = model_chunk.config.ffn_hidden_size
                fc1_split_shapes = [ffn_hidden_size, ffn_hidden_size]
        except Exception:
            pass

        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue

            trainable_params += 1
            param.param_name = name

            if "experts" in name and "shared" not in name:
                param.expert_tp = True
                tag_counts["expert_tp"] += 1

            if include_router_flag and "router.weight" in name and len(param.shape) == 2:
                param.is_router = True
                tag_counts["router"] += 1

            if "linear_qkv.weight" in name and len(param.shape) == 2:
                param.is_qkv = True
                tag_counts["qkv"] += 1

            if "linear_fc1.weight" in name and len(param.shape) == 2:
                param.is_fc1 = True
                tag_counts["fc1"] += 1
                if include_muon_extended_flags and "experts" in name:
                    param.is_moe_fc1 = True
                    tag_counts["moe_fc1"] += 1

            if include_muon_extended_flags and "linear_fc2.weight" in name and len(param.shape) == 2:
                param.is_fc2 = True
                tag_counts["fc2"] += 1
                if "experts" in name:
                    param.is_moe_fc2 = True
                    tag_counts["moe_fc2"] += 1

            if include_muon_extended_flags and (
                ("linear_proj.weight" in name)
                or ("attention.dense.weight" in name)
                or ("self_attention.linear_proj.weight" in name)
            ) and len(param.shape) == 2:
                param.is_o_proj = True
                tag_counts["o_proj"] += 1

            if include_muon_extended_flags and (
                ("embedding.word_embeddings.weight" in name)
                or ("embedding.position_embeddings.weight" in name)
            ) and len(param.shape) == 2:
                param.is_embedding = True
                tag_counts["embedding"] += 1

            if include_muon_extended_flags and (
                ("output_layer.weight" in name) or ("lm_head.weight" in name)
            ) and len(param.shape) == 2:
                param.is_lm_head = True
                tag_counts["lm_head"] += 1

            if "experts.weight1" in name or "experts.weight2" in name:
                param.is_grouped_moe = True
                tag_counts["grouped_moe"] += 1

                if include_muon_extended_flags:
                    if "experts.weight1" in name:
                        param.is_moe_fc1 = True
                        tag_counts["moe_fc1"] += 1
                    if "experts.weight2" in name:
                        param.is_moe_fc2 = True
                        tag_counts["moe_fc2"] += 1

                try:
                    param.num_local_experts = (
                        model_chunk.config.num_moe_experts // model_chunk.config.expert_model_parallel_size
                    )
                    param.moe_ffn_hidden_size = model_chunk.config.moe_ffn_hidden_size
                    param.is_gated = model_chunk.config.gated_linear_unit
                except Exception:
                    param.is_grouped_moe = False

            # Muon: include embedding/lm_head in linear bucket when explicitly vectorized.
            use_muon_for_embedding = (
                include_muon_extended_flags
                and ("embedding" in muon_vectorize or "lm_head" in muon_vectorize)
                and (getattr(param, "is_embedding", False) or getattr(param, "is_lm_head", False))
            )

            if include_muon_extended_flags:
                is_linear = (
                    (not getattr(param, "is_embedding_or_output_parameter", False) and len(param.shape) != 1)
                    or use_muon_for_embedding
                )
            else:
                is_linear = (
                    not getattr(param, "is_embedding_or_output_parameter", False)
                    and len(param.shape) == 2
                )

            if is_linear:
                linear_params.append(param)
            else:
                nonlinear_params.append(param)

    if logger is not None:
        log_single_rank(
            logger,
            logging.INFO,
            (
                f"[{optimizer_name}] param bucketing done: "
                f"trainable={trainable_params}, linear={len(linear_params)}, nonlinear={len(nonlinear_params)}, "
                f"qkv_split_shapes={qkv_split_shapes}, fc1_split_shapes={fc1_split_shapes}"
            ),
        )
        log_single_rank(
            logger,
            logging.DEBUG,
            f"[{optimizer_name}] tag counts: {tag_counts}",
        )

    return ParamBucketResult(
        linear_params=linear_params,
        nonlinear_params=nonlinear_params,
        qkv_split_shapes=qkv_split_shapes,
        fc1_split_shapes=fc1_split_shapes,
    )
