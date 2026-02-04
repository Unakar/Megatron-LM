#!/bin/bash
#SBATCH --job-name=megatron_sweep
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_barak_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --array=0  # 3 lr × 3 momentum = 9 jobs

mkdir -p logs

# Environment setup
module load cuda/12.4.1-fasrc01
module load cudnn/9.5.1.17_cuda12-fasrc01
module load gcc/12.2.0-fasrc01

source ~/.bashrc
mamba activate megatron

export CUDA_HOME=/n/sw/helmod-rocky8/apps/Core/cuda/12.4.1-fasrc01/cuda
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONNOUSERSITE=1

cd /n/home07/nabreu/megatron/Megatron-LM/hydra_wrapper

# Sweep config - any list values become sweep params
export SWEEP_CONFIG=sweeps/qwen3_1.8b_spball.yaml
export SWEEP_CMD="python train.py"

python sweep_launcher.py
