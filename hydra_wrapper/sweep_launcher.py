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
    
    # Index from SLURM or manual override
    idx = os.environ.get("SLURM_ARRAY_TASK_ID", os.environ.get("RUN_INDEX", "0"))
    try:
        idx = int(idx)
    except Exception:
        print(f"[sweep_launcher] Invalid index: {idx}", file=sys.stderr)
        sys.exit(2)

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

    # Show plan
    print(f"[sweep_launcher] config={config_path}")
    print(f"[sweep_launcher] sweep_params={list(sweep_params.keys())}")
    print(f"[sweep_launcher] fixed_params={list(fixed_params.keys())}")
    print(f"[sweep_launcher] total={total} index={idx}")
    print(f"[sweep_launcher] combo={combo}")

    # Ensure RUN_INDEX env is exported for config/wandb naming
    env = os.environ.copy()
    env["RUN_INDEX"] = str(idx)

    # Final command
    cmd = base_cmd.split() + overrides
    print("[sweep_launcher] exec:", " ".join(shlex.quote(x) for x in cmd))
    
    # Run training
    proc = subprocess.run(cmd, env=env)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
