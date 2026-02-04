# Megatron-LM Hydra Wrapper

This wrapper provides Hydra-based configuration management for Megatron-LM, enabling easy hyperparameter sweeps and experiment tracking.

## Directory Structure

```
hydra_wrapper/
├── conf/
│   ├── config.yaml           # Main config with defaults
│   ├── model/
│   │   ├── gpt_small.yaml    # 125M model
│   │   └── gpt_medium.yaml   # 350M model
│   ├── training/
│   │   └── default.yaml
│   ├── optimizer/
│   │   ├── spectral_ball.yaml
│   │   └── adam.yaml
│   ├── data/
│   │   ├── mock.yaml         # For testing
│   │   └── real.yaml         # Template for real data
│   └── distributed/
│       ├── single_node.yaml
│       └── multi_node.yaml
├── train.py                  # Main wrapper script
├── submit_sweep.sh           # Slurm sweep script
└── README.md
```

## Installation

```bash
pip install hydra-core omegaconf
```

## Usage

### Single Run
```bash
cd hydra_wrapper
python train.py
```

### Override Parameters
```bash
python train.py optimizer.lr=0.002 model=gpt_medium
```

### Switch Optimizer
```bash
python train.py optimizer=adam optimizer.lr=0.0001
```

### Hyperparameter Sweep (Hydra multirun)
```bash
python train.py --multirun optimizer.lr=0.001,0.002,0.005 optimizer.momentum=0.8,0.9
```

### Submit Slurm Array Job
```bash
# Edit submit_sweep.sh to define your sweep parameters
sbatch submit_sweep.sh
```

## Configuration Groups

### Model (`model=`)
- `gpt_small` (default): 12 layers, 768 hidden, ~125M params
- `gpt_medium`: 24 layers, 1024 hidden, ~350M params

### Optimizer (`optimizer=`)
- `spectral_ball` (default): Spectral norm constrained optimizer
- `adam`: Standard AdamW

### Data (`data=`)
- `mock` (default): Mock data for testing
- `real`: Template for real data (update paths)

### Distributed (`distributed=`)
- `single_node` (default): 4 GPUs, TP=2, PP=2
- `multi_node`: 8 GPUs/node, TP=4, PP=4

## Example Sweep Configurations

### Learning Rate Sweep
```bash
python train.py --multirun optimizer.lr=0.0005,0.001,0.002,0.005
```

### Full Grid Search
```bash
python train.py --multirun \
    optimizer.lr=0.0005,0.001,0.002 \
    optimizer.momentum=0.8,0.9,0.95 \
    optimizer.msign_steps=4,8,12
```

### Model Comparison
```bash
python train.py --multirun model=gpt_small,gpt_medium optimizer=spectral_ball,adam
```
