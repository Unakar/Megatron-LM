#!/usr/bin/env python3
"""
Hydra wrapper for Megatron-LM GPT training.

This wrapper translates Hydra configuration to Megatron-LM CLI arguments,
enabling easy hyperparameter sweeps with Hydra's multirun functionality.

Usage:
    # Single run
    python train.py

    # Override parameters
    python train.py optimizer.lr=0.002 model=gpt_medium

    # Hyperparameter sweep
    python train.py --multirun optimizer.lr=0.001,0.002,0.005 optimizer.momentum=0.8,0.9

    # With Slurm launcher (see launcher config)
    python train.py --multirun hydra/launcher=slurm optimizer.lr=0.001,0.002
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Optional

import hydra
from omegaconf import DictConfig, OmegaConf


def build_megatron_args(cfg: DictConfig) -> List[str]:
    """Convert Hydra config to Megatron-LM CLI arguments."""
    args = []

    # ==================== Model Arguments ====================
    model = cfg.model
    args.extend([
        f"--num-layers={model.num_layers}",
        f"--hidden-size={model.hidden_size}",
        f"--num-attention-heads={model.num_attention_heads}",
        f"--seq-length={model.seq_length}",
        f"--max-position-embeddings={model.max_position_embeddings}",
    ])

    # Optional model args with defaults
    if model.get("ffn_hidden_size"):
        args.append(f"--ffn-hidden-size={model.ffn_hidden_size}")
    if model.get("kv_channels"):
        args.append(f"--kv-channels={model.kv_channels}")
    if model.get("attention_backend"):
        args.append(f"--attention-backend={model.attention_backend}")
    if model.get("normalization"):
        args.append(f"--normalization={model.normalization}")
    if model.get("norm_epsilon"):
        args.append(f"--norm-epsilon={model.norm_epsilon}")
    if model.get("position_embedding_type"):
        args.append(f"--position-embedding-type={model.position_embedding_type}")
    if model.get("init_method_std"):
        args.append(f"--init-method-std={model.init_method_std}")

    # Group Query Attention (GQA)
    if model.get("group_query_attention", False):
        args.append("--group-query-attention")
        if model.get("num_query_groups"):
            args.append(f"--num-query-groups={model.num_query_groups}")

    # Rotary position embeddings
    if model.get("use_rotary_position_embeddings", False):
        args.append("--use-rotary-position-embeddings")
        if model.get("rotary_base"):
            args.append(f"--rotary-base={model.rotary_base}")

    # Dropout (explicitly set 0 if needed)
    if "attention_dropout" in model:
        args.append(f"--attention-dropout={model.attention_dropout}")
    if "hidden_dropout" in model:
        args.append(f"--hidden-dropout={model.hidden_dropout}")

    # Boolean flags
    if model.get("swiglu", False):
        args.append("--swiglu")
    if model.get("untie_embeddings_and_output_weights", False):
        args.append("--untie-embeddings-and-output-weights")
    if model.get("disable_bias_linear", False):
        args.append("--disable-bias-linear")
    if model.get("qk_layernorm", False):
        args.append("--qk-layernorm")
    if model.get("cross_entropy_loss_fusion", False):
        args.append("--cross-entropy-loss-fusion")
    if model.get("spectral_mup_init", False):
        args.append("--spectral-mup-init")
    if model.get("lecun_init", False):
        args.append("--lecun-init")
    if model.get("use_cpu_initialization", False):
        args.append("--use-cpu-initialization")
    if model.get("transformer_impl"):
        args.append(f"--transformer-impl={model.transformer_impl}")

    # Init modes
    if model.get("split_qkv_init_mode"):
        args.append(f"--split-qkv-init-mode={model.split_qkv_init_mode}")
    if model.get("embedding_init_method_std") is not None:
        args.append(f"--embedding-init-method-std={model.embedding_init_method_std}")

    # ==================== Training Arguments ====================
    training = cfg.training
    args.extend([
        f"--micro-batch-size={training.micro_batch_size}",
        f"--global-batch-size={training.global_batch_size}",
        f"--train-iters={training.train_iters}",
        f"--clip-grad={training.clip_grad}",
    ])

    # Precision
    if training.precision == "bf16":
        args.append("--bf16")
    elif training.precision == "fp16":
        args.append("--fp16")

    if training.get("no_masked_softmax_fusion", False):
        args.append("--no-masked-softmax-fusion")

    # ==================== Optimizer Arguments ====================
    optimizer = cfg.optimizer
    opt_name = optimizer.get("optimizer", optimizer.get("name", "adam"))
    args.extend([
        f"--optimizer={opt_name}",
        f"--lr={optimizer.lr}",
        f"--min-lr={optimizer.min_lr}",
        f"--weight-decay={optimizer.weight_decay}",
    ])

    # LR schedule from optimizer config
    if optimizer.get("lr_warmup_iters"):
        args.append(f"--lr-warmup-iters={optimizer.lr_warmup_iters}")
    if optimizer.get("lr_decay_style"):
        args.append(f"--lr-decay-style={optimizer.lr_decay_style}")
    if optimizer.get("lr_decay_iters"):
        args.append(f"--lr-decay-iters={optimizer.lr_decay_iters}")

    # Adam parameters (used by many optimizers)
    if optimizer.get("adam_beta1"):
        args.append(f"--adam-beta1={optimizer.adam_beta1}")
    if optimizer.get("adam_beta2"):
        args.append(f"--adam-beta2={optimizer.adam_beta2}")
    if optimizer.get("adam_eps"):
        args.append(f"--adam-eps={optimizer.adam_eps}")
    if optimizer.get("use_distributed_optimizer", False):
        args.append("--use-distributed-optimizer")
    if optimizer.get("adamw_lr_mup_scaler", False):
        args.append("--adamw-lr-mup-scaler")

    # Activation recompute/checkpointing
    if training.get("recompute_activations", False):
        args.append("--recompute-activations")
    if training.get("recompute_granularity"):
        args.append(f"--recompute-granularity={training.recompute_granularity}")

    # Spectral Ball / Spectral Ball Dist specific
    if "spectral_ball" in opt_name:
        if optimizer.get("spectral_ball_momentum"):
            args.append(f"--spectral-ball-momentum={optimizer.spectral_ball_momentum}")
        if optimizer.get("spectral_ball_use_nesterov", False):
            args.append("--spectral-ball-use-nesterov")
        if optimizer.get("spectral_ball_msign_steps"):
            args.append(f"--spectral-ball-msign-steps={optimizer.spectral_ball_msign_steps}")
        if optimizer.get("spectral_ball_power_iteration_steps"):
            args.append(f"--spectral-ball-power-iteration-steps={optimizer.spectral_ball_power_iteration_steps}")
        if optimizer.get("spectral_ball_radius_mode"):
            args.append(f"--spectral-ball-radius-mode={optimizer.spectral_ball_radius_mode}")
        if optimizer.get("spectral_ball_scale_mode"):
            args.append(f"--spectral-ball-scale-mode={optimizer.spectral_ball_scale_mode}")
        if optimizer.get("spectral_ball_solver"):
            args.append(f"--spectral-ball-solver={optimizer.spectral_ball_solver}")
        if optimizer.get("spectral_ball_solver_tolerance_f"):
            args.append(f"--spectral-ball-solver-tolerance-f={optimizer.spectral_ball_solver_tolerance_f}")
        if optimizer.get("spectral_ball_solver_max_iterations"):
            args.append(f"--spectral-ball-solver-max-iterations={optimizer.spectral_ball_solver_max_iterations}")
        if optimizer.get("spectral_ball_retract_mode"):
            args.append(f"--spectral-ball-retract-mode={optimizer.spectral_ball_retract_mode}")
        if optimizer.get("spectral_ball_qkv_split_mode"):
            args.append(f"--spectral-ball-qkv-split-mode={optimizer.spectral_ball_qkv_split_mode}")

    # Muon (standalone) specific
    if opt_name == "muon":
        if optimizer.get("muon_momentum"):
            args.append(f"--muon-momentum={optimizer.muon_momentum}")
        if optimizer.get("muon_use_nesterov", False):
            args.append("--muon-use-nesterov")
        if optimizer.get("muon_scale_mode"):
            args.append(f"--muon-scale-mode={optimizer.muon_scale_mode}")
        if optimizer.get("muon_num_ns_steps"):
            args.append(f"--muon-num-ns-steps={optimizer.muon_num_ns_steps}")

    # MuonHyperball specific
    if "muon_hyperball" in opt_name:
        if optimizer.get("muon_hyperball_momentum"):
            args.append(f"--muon-hyperball-momentum={optimizer.muon_hyperball_momentum}")
        if optimizer.get("muon_hyperball_use_nesterov", False):
            args.append("--muon-hyperball-use-nesterov")
        if optimizer.get("muon_hyperball_msign_steps"):
            args.append(f"--muon-hyperball-msign-steps={optimizer.muon_hyperball_msign_steps}")
        if optimizer.get("muon_hyperball_radius_mode"):
            args.append(f"--muon-hyperball-radius-mode={optimizer.muon_hyperball_radius_mode}")
        if optimizer.get("muon_hyperball_scale_mode"):
            args.append(f"--muon-hyperball-scale-mode={optimizer.muon_hyperball_scale_mode}")
        if optimizer.get("muon_hyperball_qkv_split_mode"):
            args.append(f"--muon-hyperball-qkv-split-mode={optimizer.muon_hyperball_qkv_split_mode}")

    # HyperballAdam specific
    if "hyperball_adam" in opt_name:
        if optimizer.get("hyperball_adam_beta1"):
            args.append(f"--hyperball-adam-beta1={optimizer.hyperball_adam_beta1}")
        if optimizer.get("hyperball_adam_beta2"):
            args.append(f"--hyperball-adam-beta2={optimizer.hyperball_adam_beta2}")
        if optimizer.get("hyperball_adam_eps"):
            args.append(f"--hyperball-adam-eps={optimizer.hyperball_adam_eps}")
        if optimizer.get("hyperball_adam_bias_correction", True):
            args.append("--hyperball-adam-bias-correction")

    # ==================== Data Arguments ====================
    data = cfg.data
    use_separate_paths = False
    if data.use_mock:
        args.extend([
            "--mock-data",
            f"--vocab-size={cfg.model.vocab_size}",
            "--tokenizer-type=NullTokenizer",
        ])
    else:
        # Handle separate train/valid paths or single data path
        if data.get("train_data_path") and data.get("valid_data_path"):
            args.extend([
                f"--train-data-path={data.train_data_path}",
                f"--valid-data-path={data.valid_data_path}",
            ])
            use_separate_paths = True
        elif data.get("data_path"):
            args.append(f"--data-path={data.data_path}")
        
        # Tokenizer configuration
        args.append(f"--tokenizer-type={data.tokenizer_type}")
        
        # HuggingFaceTokenizer uses tokenizer_model, others use vocab_file/merge_file
        if data.tokenizer_type == "HuggingFaceTokenizer":
            args.append(f"--tokenizer-model={data.tokenizer_model}")
        else:
            if data.get("vocab_file"):
                args.append(f"--vocab-file={data.vocab_file}")
            if data.get("merge_file"):
                args.append(f"--merge-file={data.merge_file}")

        # Data loading options
        if data.get("data_cache_path"):
            args.append(f"--data-cache-path={data.data_cache_path}")
        if data.get("num_dataset_builder_threads"):
            args.append(f"--num-dataset-builder-threads={data.num_dataset_builder_threads}")
        if data.get("num_workers"):
            args.append(f"--num-workers={data.num_workers}")
        if data.get("no_mmap_bin_files", False):
            args.append("--no-mmap-bin-files")
        if data.get("distributed_timeout_minutes"):
            args.append(f"--distributed-timeout-minutes={data.distributed_timeout_minutes}")
    
    # Only add split when using single data_path (incompatible with train/valid paths)
    if not use_separate_paths and data.get("split"):
        args.append(f"--split={data.split}")

    # ==================== Distributed Arguments ====================
    dist = cfg.distributed
    args.extend([
        f"--tensor-model-parallel-size={dist.tensor_model_parallel_size}",
        f"--pipeline-model-parallel-size={dist.pipeline_model_parallel_size}",
    ])
    if dist.sequence_parallel:
        args.append("--sequence-parallel")

    # ==================== Logging Arguments ====================
    training = cfg.training
    args.extend([
        f"--save={cfg.paths.checkpoint_dir}",
        f"--load={cfg.paths.checkpoint_dir}",
        f"--tensorboard-dir={cfg.paths.tensorboard_dir}",
        f"--log-interval={training.get('log_interval', 10)}",
        f"--hidden-state-log-interval={training.get('hidden_state_log_interval', 100)}",
        f"--save-interval={training.get('save_interval', 1000)}",
    ])
    if training.get("save_retain_interval"):
        args.append(f"--save-retain-interval={training.save_retain_interval}")
    args.extend([
        f"--eval-interval={training.get('eval_interval', 500)}",
        f"--eval-iters={training.get('eval_iters', 10)}",
    ])

    # Early stopping on target validation loss
    if training.get("target_val_loss") is not None:
        args.append(f"--target-val-loss={training.target_val_loss}")
    if training.get("target_val_loss_patience"):
        args.append(f"--target-val-loss-patience={training.target_val_loss_patience}")

    # Exponential Weight Averaging (EWA)
    if training.get("ewa_decay") is not None:
        args.append(f"--ewa-decay={training.ewa_decay}")
    if training.get("ewa_start_iter"):
        args.append(f"--ewa-start-iter={training.ewa_start_iter}")

    # Boolean logging flags
    if training.get("log_throughput", True):
        args.append("--log-throughput")
    if training.get("log_params_norm", False):
        args.append("--log-params-norm")
    if training.get("log_num_zeros_in_grad", False):
        args.append("--log-num-zeros-in-grad")
    if training.get("log_validation_ppl_to_tensorboard", False):
        args.append("--log-validation-ppl-to-tensorboard")
    if training.get("log_timers_to_tensorboard", False):
        args.append("--log-timers-to-tensorboard")
    if training.get("log_memory_to_tensorboard", False):
        args.append("--log-memory-to-tensorboard")
    if training.get("log_world_size_to_tensorboard", False):
        args.append("--log-world-size-to-tensorboard")

    # Checkpoint format options
    if training.get("ckpt_format"):
        args.append(f"--ckpt-format={training.ckpt_format}")
    if training.get("no_save_optim", False):
        args.append("--no-save-optim")
    if training.get("async_save", False):
        args.append("--async-save")

    # Detailed per-module logging
    if training.get("log_per_module_update_rms", False):
        args.append("--log-per-module-update-rms")
    if training.get("log_per_module_grad_rms", False):
        args.append("--log-per-module-grad-rms")
    
    # Log hidden states for specific modules
    log_hidden_states = training.get("log_hidden_states", [])
    if log_hidden_states:
        args.append("--log-hidden-states")
        args.extend(log_hidden_states)
    
    # Log parameter statistics for specific modules
    log_params = training.get("log_params", [])
    if log_params:
        args.append("--log-params")
        args.extend(log_params)

    # Benchmark evaluation
    if training.get("benchmark_eval", False):
        args.append("--benchmark-eval")
    if training.get("benchmark_interval"):
        args.append(f"--benchmark-interval={training.benchmark_interval}")
    benchmark_tasks = training.get("benchmark_tasks", [])
    if benchmark_tasks:
        # Join tasks with comma for --benchmark-tasks format
        args.append(f"--benchmark-tasks={','.join(benchmark_tasks)}")
    
    if training.get("benchmark_sequence_length"):
        args.append(f"--benchmark-sequence-length={training.benchmark_sequence_length}")

    # ==================== nGPT Weight Normalization ====================
    ngpt = cfg.get("ngpt", {})
    if ngpt and ngpt.get("weight_norm", False):
        args.append("--ngpt-weight-norm")
        if ngpt.get("weight_norm_forward", False):
            args.append("--ngpt-weight-norm-forward")
        if ngpt.get("weight_norm_eps"):
            args.append(f"--ngpt-weight-norm-eps={ngpt.weight_norm_eps}")
        if ngpt.get("weight_norm_targets"):
            targets = ngpt.weight_norm_targets
            args.append("--ngpt-weight-norm-targets")
            # OmegaConf ListConfig needs explicit iteration
            if hasattr(targets, '__iter__') and not isinstance(targets, str):
                for t in targets:
                    args.append(str(t))
            else:
                args.append(str(targets))
        if ngpt.get("weight_norm_log_interval"):
            args.append(f"--ngpt-weight-norm-log-interval={ngpt.weight_norm_log_interval}")

    # W&B
    if cfg.wandb.enabled:
        args.append(f"--wandb-project={cfg.wandb.project}")
        args.append(f"--wandb-exp-name={cfg.wandb.name}")
        if cfg.wandb.entity:
            args.append(f"--wandb-entity={cfg.wandb.entity}")
        if cfg.wandb.get("save_dir"):
            args.append(f"--wandb-save-dir={cfg.wandb.save_dir}")

    return args


def get_torchrun_args(cfg: DictConfig) -> List[str]:
    """Build torchrun distributed arguments."""
    dist = cfg.distributed

    # Get Slurm environment variables if available
    node_rank = int(os.environ.get("SLURM_NODEID", 0))
    num_nodes = int(os.environ.get("SLURM_NNODES", dist.num_nodes))

    if "SLURM_JOB_NODELIST" in os.environ:
        # Get master address from Slurm
        result = subprocess.run(
            ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]],
            capture_output=True,
            text=True,
        )
        master_addr = result.stdout.strip().split("\n")[0]
    else:
        master_addr = "localhost"

    master_port = os.environ.get("MASTER_PORT", "29500")

    return [
        f"--nproc_per_node={dist.gpus_per_node}",
        f"--nnodes={num_nodes}",
        f"--node_rank={node_rank}",
        f"--master_addr={master_addr}",
        f"--master_port={master_port}",
    ]


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> Optional[float]:
    """Main training function."""
    # Print config
    print("=" * 60)
    print("Megatron-LM Training with Hydra")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    # Create output directories
    Path(cfg.paths.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.tensorboard_dir).mkdir(parents=True, exist_ok=True)

    # Build command
    megatron_dir = Path(cfg.paths.megatron_dir)
    pretrain_script = megatron_dir / "pretrain_gpt.py"

    torchrun_args = get_torchrun_args(cfg)
    megatron_args = build_megatron_args(cfg)

    cmd = ["torchrun"] + torchrun_args + [str(pretrain_script)] + megatron_args

    print("\nCommand:")
    print(" ".join(cmd))
    print("\n" + "=" * 60)

    # Set environment variables
    env = os.environ.copy()
    env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    env["PYTHONPATH"] = f"{megatron_dir}:{env.get('PYTHONPATH', '')}"
    if cfg.wandb.get("mode"):
        env["WANDB_MODE"] = str(cfg.wandb.mode)

    # Run training
    result = subprocess.run(cmd, env=env)

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with return code {result.returncode}")

    return None


if __name__ == "__main__":
    main()
