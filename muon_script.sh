#!/bin/bash

# 这行的意思是：如果脚本中的任何命令返回非零状态（即出错），脚本会立即退出（-e）；同时如果管道中的任何一个命令失败，整个管道会被视为失败（-o pipefail）。
set -eo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

# 激活 megatron 环境
source /volume/pt-train/users/rbliu/miniconda3/bin/activate base
conda activate megatron

TOTAL_TOKENS=100000000000 # 大概100B

# Multi-node Arguments
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${PET_NNODES:-"1"}
NODE_RANK=${PET_NODE_RANK:-"0"}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

GLOBAL_BATCH=1024
TRAIN_ITER=$((TOTAL_TOKENS / GLOBAL_BATCH / 4096))

# Training Arguments
TP_SIZE=${TP_SIZE:-1}   # 张量并行（Tensor Model Parallel）大小
PP_SIZE=${PP_SIZE:-1}   # 流水线并行（Pipeline Model Parallel）大小
LOG_INTERVAL=${LOG_INTERVAL:-1}

REPO_PATH="/volume/pt-train/users/rbliu/checkpoint/Megatron-LM"
JOB_NAME=muon-qwen-750M
TENSORBOARD_PATH="${REPO_PATH}/tensorboard/${JOB_NAME}"
CHECKPOINT_PATH="/volume/pt-data/data/rbliu/checkpoints/${JOB_NAME}"
WANDB_PATH="${REPO_PATH}/wandb/${JOB_NAME}"

# 如果目录存在则删除
[ -d "$TENSORBOARD_PATH" ] && rm -rf "$TENSORBOARD_PATH"
# [ -d "$CHECKPOINT_PATH" ] && rm -rf "$CHECKPOINT_PATH"
# [ -d "$WANDB_PATH" ] && rm -rf "$WANDB_PATH"

LOG_DIR="/volume/pt-train/users/rbliu/github/Megatron-LM/logs"
mkdir -p $LOG_DIR

SAVE_LOG="${LOG_DIR}/${JOB_NAME}_save.log"
# 若 log 文件已存在则删除
# [ -f "$SAVE_LOG" ] && rm -f "$SAVE_LOG"

LOAD_LOG="${LOG_DIR}/${JOB_NAME}_load.log"
# 若 log 文件已存在则删除
[ -f "$LOAD_LOG" ] && rm -f "$LOAD_LOG"

mkdir -p $TENSORBOARD_PATH
mkdir -p $CHECKPOINT_PATH
mkdir -p $WANDB_PATH

PRETRAIN_SCRIPT="/volume/pt-train/users/rbliu/github/Megatron-LM/pretrain_gpt.py"

BASE_PATH="${BASE_PATH:-/volume/pt-train/users/rbliu/dataset/train_dataset/qwen}"
DATA_PATH=""

while IFS= read -r file; do
    common_prefix=${file%".bin"}
    DATA_PATH+="1 ${common_prefix} "
done < <(find "$BASE_PATH" -type f -path "**.bin")

DATA_PATH_CACHE="/volume/pt-train/users/rbliu/dataset/dataset_cache/qwen"

DATA_ARGS=(
    --tokenizer-model /volume/pt-train/users/rbliu/model/Qwen3-0.6B-Base
    --tokenizer-type HuggingFaceTokenizer
    --data-path $DATA_PATH
    --data-cache-path ${DATA_PATH_CACHE}
    --train-iters $TRAIN_ITER
    --split 99,1,0
    --num-dataset-builder-threads 128
    --num-workers 16
    --no-mmap-bin-files
    --distributed-timeout-minutes 60
)

TRAINING_ARGS=(
    --lr 5e-3
    --min-lr 5e-4
    --lr-warmup-iters 500
    --lr-decay-style cosine
    --lr-decay-iters $TRAIN_ITER
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --clip-grad 1.0
    --weight-decay 0.1
    --optimizer muon_ball_dist
    --muon-ball-momentum 0.9
    --muon-ball-use-nesterov
    --muon-ball-msign-steps 8
    --muon-ball-radius-mode spectral_mup
    --muon-ball-scale-mode spectral_mup
    --muon-ball-power-iteration-steps 100
    --muon-ball-retract-mode hard
    --muon-ball-qkv-split-mode head
    # --recompute-activations
    # --recompute-granularity full
)


MODEL_ARGS=(
    --num-layers 28
    --hidden-size 1024
    --ffn-hidden-size 3072
    --group-query-attention
    --num-attention-heads 16
    --num-query-groups 8
    --norm-epsilon 1e-6
    --kv-channels 128
    --seq-length 4096
    --max-position-embeddings 40960
    --attention-dropout 0
    --hidden-dropout 0
    --bf16
    --use-rotary-position-embeddings
    --rotary-base 1000000
    --swiglu
    --untie-embeddings-and-output-weights
    --normalization RMSNorm
    --qk-layernorm
    --cross-entropy-loss-fusion
    --disable-bias-linear
    --transformer-impl transformer_engine
    --attention-backend fused
    --init-method-std 0.02
    --split-qkv-init-mode head
    --spectral-mup-init
    --use-cpu-initialization
)

PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP_SIZE}
    --pipeline-model-parallel-size ${PP_SIZE}
    --micro-batch-size 8
    --global-batch-size ${GLOBAL_BATCH}
)

LOGGER_ARGS=(
    --log-params-norm
    --log-throughput
    --log-interval 1
    --log-params-norm
    --log-num-zeros-in-grad
    --log-validation-ppl-to-tensorboard
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard
    --log-world-size-to-tensorboard
    --tensorboard-dir ${TENSORBOARD_PATH}
    --log-per-module-update-rms
    --log-per-module-grad-rms
    --log-hidden-states embeddings input_layernorm attention::linear_qkv attention::linear_q attention::linear_k attention::linear_v attention::core_attention attention::o_proj pre_mlp_layernorm mlp
    --log-params attention::linear_qkv attention::o_proj mlp::linear_fc1 mlp::linear_fc2 input_layernorm pre_mlp_layernorm embedding lm_head

)

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

WANDB_ARGS=(
    --wandb-project optimizer_arena_v2
    --wandb-exp-name $JOB_NAME
    --wandb-save-dir ${WANDB_PATH}
)

# CKPT_ARGS=(
#     --ckpt-format "torch_dist"
#     --save-interval 50
#     --exit-interval 50
#     # --no-save-optim
#     # --async-save
#     --save $CHECKPOINT_PATH
#     --load $CHECKPOINT_PATH
# )

# {
#     torchrun \
#         ${DISTRIBUTED_ARGS[@]} \
#         $PRETRAIN_SCRIPT \
#         ${DATA_ARGS[@]} \
#         ${MODEL_ARGS[@]} \
#         ${TRAINING_ARGS[@]} \
#         ${PARALLEL_ARGS[@]} \
#         ${CKPT_ARGS[@]} \
#         ${LOGGER_ARGS[@]} \
#         ${WANDB_ARGS[@]} \
#         --eval-interval 10 \
#         --eval-iters 1
# } 2>&1 | tee -a "$SAVE_LOG"

CKPT_ARGS=(
    --ckpt-format "torch_dist"
    --save-interval 10
    --exit-interval 100
    # --no-save-optim
    # --async-save
    --save $CHECKPOINT_PATH
    --load $CHECKPOINT_PATH
)

{
    torchrun \
        ${DISTRIBUTED_ARGS[@]} \
        $PRETRAIN_SCRIPT \
        ${DATA_ARGS[@]} \
        ${MODEL_ARGS[@]} \
        ${TRAINING_ARGS[@]} \
        ${PARALLEL_ARGS[@]} \
        ${CKPT_ARGS[@]} \
        ${LOGGER_ARGS[@]} \
        ${WANDB_ARGS[@]} \
        --eval-interval 10 \
        --eval-iters 1
} 2>&1 | tee -a "$LOAD_LOG"