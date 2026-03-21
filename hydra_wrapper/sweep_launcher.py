#!/usr/bin/env python3
"""
Sweep launcher for Megatron-LM Hydra wrapper.

Automatically detects sweep parameters from config.yaml - any parameter 
that is a list will be swept over. Single values are used as-is.

Environment variables:
    SWEEP_CONFIG: Path to config YAML (default: conf/config.yaml)
    SWEEP_CMD: Base command to run (default: python train.py)
    SLURM_ARRAY_TASK_ID or RUN_INDEX: Which combination to run

Usage:
    # In your Slurm script:
    SWEEP_CONFIG=conf/config.yaml python sweep_launcher.py
    
    # Or specify a sweep-specific config:
    SWEEP_CONFIG=sweeps/lr_sweep.yaml python sweep_launcher.py
"""

import os
import sys
import itertools
import yaml
import subprocess
import shlex

# Parameters that are natively list-valued (not sweep params even if they contain a list)
NATIVE_LIST_PARAMS = {
    'log_hidden_states',
    'log_params',
    'benchmark_tasks',
    'training.log_hidden_states',
    'training.log_params',
    'training.benchmark_tasks',
}


def is_native_list_param(key):
    """Check if a parameter key should be treated as a native list (not a sweep param)."""
    # Check if the key or its last component matches known native list params
    return key in NATIVE_LIST_PARAMS or key.split('.')[-1] in NATIVE_LIST_PARAMS


def find_sweep_params(d, parent_key='', sep='.'):
    """
    Recursively find all list-valued parameters (these are sweep params).
    Returns dict of {dotted.key: [values]} for sweep params only.
    Excludes parameters that are natively list-valued.
    """
    sweep_params = {}
    for k, v in d.items():
        full_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            sweep_params.update(find_sweep_params(v, full_key, sep))
        elif isinstance(v, list) and not is_native_list_param(full_key):
            sweep_params[full_key] = v
    return sweep_params


def flatten_fixed_params(d, parent_key='', sep='.'):
    """
    Recursively find all non-list parameters (fixed values).
    Returns dict of {dotted.key: value} for fixed params only.
    Also includes native list params as fixed values.
    """
    fixed_params = {}
    for k, v in d.items():
        full_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            fixed_params.update(flatten_fixed_params(v, full_key, sep))
        elif not isinstance(v, list):
            fixed_params[full_key] = v
        elif is_native_list_param(full_key):
            # Native list params are fixed, not swept
            fixed_params[full_key] = v
    return fixed_params


def cartesian_product(grid_dict):
    """Return list of dicts for all combinations."""
    if not grid_dict:
        return [{}]
    keys = list(grid_dict.keys())
    vals = [grid_dict[k] for k in keys]
    combos = []
    for prod in itertools.product(*vals):
        combos.append({k: v for k, v in zip(keys, prod)})
    return combos


def main():
    # Inputs from environment
    config_path = os.environ.get("SWEEP_CONFIG", "conf/config.yaml")
    base_cmd = os.environ.get("SWEEP_CMD", "python train.py")
    
    # Resume options
    resume_from_job = os.environ.get("RESUME_FROM_JOB")        # Option B: SLURM job ID
    resume_checkpoint_path = os.environ.get("RESUME_CHECKPOINT_PATH")  # Option A: explicit path
    resume_wandb_job_id = os.environ.get("RESUME_WANDB_JOB_ID")       # Option A: WandB resume

    # Index from SLURM or manual override
    idx = os.environ.get("SLURM_ARRAY_TASK_ID", os.environ.get("RUN_INDEX", "0"))
    try:
        idx = int(idx)
    except Exception:
        print(f"[sweep_launcher] Invalid index: {idx}", file=sys.stderr)
        sys.exit(2)

    # Current SLURM identifiers
    current_job = os.environ.get("SLURM_ARRAY_JOB_ID",
                                 os.environ.get("SLURM_JOB_ID", "local"))
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", str(idx))

    # Effective job ID: use resume job if resuming, otherwise current
    effective_job = resume_from_job if resume_from_job else current_job

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}
    
    # Find sweep params (lists) and fixed params (single values)
    sweep_params = find_sweep_params(config)
    
    if not sweep_params:
        print("[sweep_launcher] No sweep params found (no lists in config).", file=sys.stderr)
        print("[sweep_launcher] Running single job with fixed config.", file=sys.stderr)
        combos = [{}]
    else:
        combos = cartesian_product(sweep_params)
    
    total = len(combos)

    if not (0 <= idx < total):
        print(f"[sweep_launcher] Index {idx} out of range [0, {total-1}].", file=sys.stderr)
        sys.exit(1)

    combo = combos[idx]

    # Build Hydra-style overrides: key=value
    # Include both fixed params and the selected sweep combo
    fixed_params = flatten_fixed_params(config)
    
    # Format list values for Hydra (e.g., [a, b] -> "[a,b]")
    def format_value(v):
        if isinstance(v, list):
            return "[" + ",".join(str(x) for x in v) + "]"
        return v
    
    overrides = [f"{k}={format_value(v)}" for k, v in fixed_params.items()]
    overrides += [f"{k}={v}" for k, v in combo.items()]

    # Generate unique experiment name from sweep combo
    combo_parts = []
    for k, v in sorted(combo.items()):
        short_key = k.split('.')[-1]  # e.g., optimizer.lr -> lr
        combo_parts.append(f"{short_key}-{v}")
    combo_suffix = "_".join(combo_parts) if combo_parts else f"run{idx}"

    # Use wandb.name or experiment_name as base, append combo suffix
    base_name = fixed_params.get('wandb.name',
                                 fixed_params.get('experiment_name', 'sweep'))
    unique_name = f"{base_name}_{combo_suffix}"
    overrides.append(f"experiment_name={unique_name}")

    # Also make wandb.name unique if present
    if 'wandb.name' in fixed_params:
        overrides = [o for o in overrides if not o.startswith('wandb.name=')]
        overrides.append(f"wandb.name={unique_name}")

    # Show plan
    print(f"[sweep_launcher] config={config_path}")
    print(f"[sweep_launcher] sweep_params={list(sweep_params.keys())}")
    print(f"[sweep_launcher] fixed_params={list(fixed_params.keys())}")
    print(f"[sweep_launcher] total={total} index={idx}")
    print(f"[sweep_launcher] combo={combo}")
    print(f"[sweep_launcher] experiment_name={unique_name}")

    # ==================== Checkpoint Dir & Resume Logic ====================
    # Use {output_dir}/checkpoints/{job_id}_{task_id} convention for checkpoint dirs.
    # This makes it easy to resume by specifying the original SLURM job ID.
    output_dir = fixed_params.get('paths.output_dir', '')

    if resume_checkpoint_path:
        # Option A: explicit checkpoint path
        checkpoint_dir = resume_checkpoint_path
        print(f"[sweep_launcher] RESUME (Option A): checkpoint_dir={checkpoint_dir}")
    elif output_dir:
        # Standard convention: {output_dir}/checkpoints/{effective_job}_{task_id}
        checkpoint_dir = f"{output_dir}/checkpoints/{effective_job}_{task_id}"
        if resume_from_job:
            print(f"[sweep_launcher] RESUME (Option B): job={resume_from_job} "
                  f"checkpoint_dir={checkpoint_dir}")
    else:
        checkpoint_dir = None

    if checkpoint_dir:
        # Remove any existing paths.checkpoint_dir overrides
        overrides = [o for o in overrides if not o.startswith('paths.checkpoint_dir=')]
        overrides.append(f"paths.checkpoint_dir={checkpoint_dir}")

    # For resume: replace SLURM_JOB_ID resolver references in overrides so that
    # experiment_name, wandb.name, etc. use the original job ID (not the current one).
    # This ensures WandB run ID derivation produces the same ID as the original run.
    original_job_id = resume_from_job or resume_wandb_job_id
    if original_job_id:
        overrides = [o.replace('${oc.env:SLURM_JOB_ID,local}', original_job_id)
                       .replace('${oc.env:SLURM_ARRAY_JOB_ID,${oc.env:SLURM_JOB_ID,local}}',
                                original_job_id)
                     for o in overrides]

    # Ensure RUN_INDEX env is exported for config/wandb naming
    env = os.environ.copy()
    env["RUN_INDEX"] = str(idx)

    # Set WandB resume env var so global_vars.py uses the original job ID
    if resume_from_job:
        env["WANDB_RESUME_JOB_ID"] = resume_from_job
    elif resume_wandb_job_id:
        env["WANDB_RESUME_JOB_ID"] = resume_wandb_job_id

    # Final command
    extra_args = sys.argv[1:]
    cmd = base_cmd.split() + overrides + extra_args
    print("[sweep_launcher] exec:", " ".join(shlex.quote(x) for x in cmd))
    
    # Run training
    proc = subprocess.run(cmd, env=env)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
