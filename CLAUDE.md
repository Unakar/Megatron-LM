# CLAUDE.md

This file provides guidance to Droid (claude.ai/code) when working with code in this repository.

## 仓库概述

这是 NVIDIA Megatron-LM 的开发分支，用于大规模 Transformer 模型训练。当前版本 0.12.0，基于 Apache 2.0 许可证。

## 常用命令

### 训练脚本示例 (muon_script.sh)
```bash
# 使用 Muon Ball 优化器训练 Qwen 750M 模型
bash muon_script.sh
```

## 代码架构

### 核心目录结构
- `megatron/core/` - Megatron Core 核心库
- `megatron/core/optimizer/` - 优化器实现
- `emerging_optimizers/` - 新型优化器库（独立包）
- `tests/unit_tests/` - 单元测试
- `tests/functional_tests/` - 功能测试
- `examples/` - 各类模型训练脚本示例

### emerging_optimizers 模块

实验性优化器库：

**正交化优化器 (orthogonalized_optimizers/)**
- `muon.py` - Muon 优化器：Newton-Schulz 迭代正交化
- `muon_ball.py` - MuonBall 优化器：Spectral Ball 简化版（λ=0）
- `spectral_ball.py` - Spectral Ball 优化器：完整谱约束优化

**标量优化器 (scalar_optimizers/)**
- `adam.py`, `ademamix.py`, `lion.py`, `signum.py`, `laprop.py`

### Megatron 优化器集成

`megatron/core/optimizer/` 关键文件：
- `muon_ball_optimizer.py` - MuonBall 与 Megatron 的集成
- `spectral_ball_optimizer.py` - SpectralBall 与 Megatron 的集成
- `optimizer_config.py` - 优化器配置类
- `layer_wise_optimizer.py` - 层级分布式优化器

## 优化器选择指南

| 优化器 | 适用场景 | 特点 |
|--------|----------|------|
| `adam` | 通用 | 标准 AdamW |
| `muon_ball_dist` | 大规模预训练 | 谱约束 + 快速（无 λ 求解） |
| `spectral_ball_dist` | 研究实验 | 完整谱约束（有 λ 求解） |

## 代码规范

- 行长度限制：100 字符
- 格式化工具：black 24.x, isort 5.13, ruff 0.9.x
- Python 版本：>= 3.10
- 测试框架：pytest 8.3.5

## LayerWiseDistributedOptimizer 架构

### 类关系
```
LayerWiseDistributedOptimizer (继承 ChainedOptimizer)
  └── Float16OptimizerWithFloat16Params (MuonBall)
  └── Float16OptimizerWithFloat16Params (Adam)
```

### 参数分配规则

`muon_ball_dist` 模式下根据参数类型分配:
- **MuonBall**: 2D 线性层权重 (`len(param.shape) == 2` 且非 embedding)
- **Adam**: 其他参数 (1D layernorm + embedding)

### 参数分片机制

`shard_params()` 使用 round-robin 将参数分配到 DP ranks：
- 分片前: 每个参数一个 param_group
- 分片后: 每个 rank 约 14 个本地参数（112 个 groups 中大多数为空）

**关键数据结构**:
- `global_param_groups`: 分片前完整的参数组列表
- `global_float16_groups`: 全局 float16 参数结构
- `float16_groups`: 本地参数的 float16 版本
- `fp32_from_float16_groups`: 本地参数的 fp32 master weights

### Checkpoint 保存/加载机制

**保存流程**:
1. `sharded_state_dict()` 生成 112 个 groups（14 个 ShardedTensor + 98 个空列表）
2. `extract_sharded_base()` 分离：sharded_part 有 14 个 tensor，common 有 98 个空列表
3. checkpoint 系统保存到 sharded files 和 common.pt

**加载流程**:
1. `_sync_global_float16_structure()` 构建全局结构
2. `sharded_state_dict(is_loading=True)` 生成匹配结构
3. `merge()` 合并 common（98 个空列表）和 loaded（14 个 tensor），返回 loaded
4. `load_state_dict()` 检测本地映射场景，直接 flatten 并按位置映射

### 关键修复

详见 `BUILD.md`：
- `merge()` 添加空列表合并处理
- `apply_factory_merges()` 跳过越界索引
- `load_state_dict()` 修复 fp32_from_fp16_params 映射逻辑

## 开发任务

- 运行 `muon_script.sh` 输出日志到 `logs/muon-qwen-750M_load.log`
- 根据日志修改代码，确保 checkpoint 加载后训练正常
- 修改记录保存到 `BUILD.md`
- 不修改 `emerging_optimizers` 中的文件
- 只修改加载逻辑，不修改保存逻辑
- 运行muon_script.sh时会输出日志到logs/muon-qwen-750M_load.log
- 你需要根据log中反应的错误和结果，修改代码，确保可以正常运行和加载优化器的状态
- 你需要首先分析和明确问题出在保存优化器状态还是加载优化器状态上面，然后再进行对应的修复
- 将所有对仓库理解有帮助、有利于你修复bug的信息和记忆及时更新和保存到CLAUDE.md
- 将对代码的修改和对应的改进思路及时更新和保存到BUILD.md，需要总结所有文件的变更和理由
- 要求再次运行muon_script.sh后可以成功加载模型状态和优化器状态进行训练
- 你不能也不需要运行脚本，会有其他人运行muon_script.sh，你只需要观察日志中结果，并根据结果修改代码
- 完成对代码的修改后及时更新CLAUDE.md和BUILD.md，确保CLAUDE.md和BUILD.md是最新的
- 你需要确保你的修改不涉及到emerging_optimizers里面的文件，尽量不要变动过多文件
- 你不能修改checkpoint的保存逻辑，只能修改加载逻辑，要求修改是最小的

## 分布式检查点结构

详见 `CHECKPOINT.md`，包含：
- 模型权重 (BF16)
- Muon 优化器状态 (momentum_buffer)
- AdamW 优化器状态 (exp_avg, exp_avg_sq)
- FP32 Master Weights
- 按 DP rank 分布在 8 个 shard 文件中
