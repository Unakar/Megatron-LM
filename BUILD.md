# BUILD.md - LayerWiseDistributedOptimizer Checkpoint 加载修复

## 问题概述

`LayerWiseDistributedOptimizer` 分片参数后，checkpoint 加载出现不匹配问题：
- 每个 DP rank 只持有部分参数（如 14/112）
- checkpoint 包含所有 rank 的数据
- 加载时 `sharded_state_dict` 结构与 checkpoint 不匹配

## 参数分布

**MuonBall**: 2D 线性层权重（`linear_qkv`, `linear_proj`, `linear_fc1`, `linear_fc2`）
**Adam**: 1D layernorm 权重 + embedding

**Round-robin 分配** (8 个 DP ranks, 112 个 MuonBall 参数):
- Rank 0: 参数 0, 8, 16, ... (14 个)
- Rank 1: 参数 1, 9, 17, ... (14 个)
- ...

## 关键数据结构

- `global_param_groups`: 分片前完整参数组列表
- `global_float16_groups`: 全局 float16 参数组结构
- `float16_groups`: 本地 float16 参数（112 个 groups，大多数为空）
- `fp32_from_float16_groups`: 本地 fp32 master weights

## 修复的问题列表

### Error 6: merge() 列表长度不匹配 (98 vs 14)

**原因**: 保存时 `fp32_from_fp16_params` 有 112 个 groups（14 个 ShardedTensor + 98 个空列表），加载时 common.pt 有 98 个空列表，sharded files 有 14 个 tensor，merge() 失败。

**修复**: `dict_utils.py` 的 `merge()` 添加特殊处理，当 x1 全是空列表、x2 有实际数据时返回 x2。

### Error 7: apply_factory_merges() 越界访问

**原因**: optimizer 继承 model 的 ShardedTensorFactory，factory 使用全局索引，但 merge() 返回 14 个元素的短列表。

**修复**: `mapping.py` 的 `apply_factory_merges()` 跳过越界索引（非本地参数没有数据）。

### Error 10: fp32_from_fp16_params 映射错误 (Loss/Grad Norm Spike)

**症状**: checkpoint 加载成功，但 iteration 52 出现 loss spike (6.88 → 9.32) 和 grad_norm spike (2.31 → 38.6)

**原因**: `load_state_dict()` 中判断条件错误：
```python
# 错误代码
num_local_groups = len(self.float16_groups)  # = 112 (groups 数量，非参数数量)
num_global_groups = len(self.global_float16_groups)  # = 112
num_saved_groups = len(saved_groups)  # = 14

# 条件永远不满足 (112 == 112)
if num_saved_groups == num_local_groups and num_saved_groups < num_global_groups:
    # 直接映射
else:
    # 走这个分支，zip(global[0:14], saved[0:14]) 错误映射
```

- `saved_groups` 按本地顺序：`[local_0, local_1, ...]` = `[global_0, global_8, ...]`
- `global_float16_groups` 按全局顺序：`[global_0, global_1, global_2, ...]`
- `zip()` 错误地把 `saved_groups[1]` (global_8 的数据) 映射到 `global_float16_groups[1]` (global_1)

**修复**: 改用实际参数数量判断：
```python
num_local_params = sum(len(g) for g in self.float16_groups)  # = 14
num_global_params = sum(len(g) for g in self.global_float16_groups)  # = 112

# 条件满足: 14 == 14 and 14 < 112
if num_saved_groups == num_local_params and num_saved_groups < num_global_params:
    # flatten 并按本地顺序直接映射
    fp32_params_flat = [fp32_p for g in self.fp32_from_float16_groups for fp32_p in g]
    saved_params_flat = [sp for sg in saved_groups for sp in sg]
    for fp32_param, saved_param in zip(fp32_params_flat, saved_params_flat):
        fp32_param.data.copy_(saved_param.data)
```

## 文件变更汇总

| 文件 | 变更 |
|------|------|
| `megatron/core/optimizer/layer_wise_optimizer.py` | `shard_params()` 保存全局结构；`_sync_global_float16_structure()` 计算 global_float16_groups |
| `megatron/core/optimizer/optimizer.py` | `sharded_state_dict()` 使用全局结构生成；`load_state_dict()` 修复 fp32_from_fp16_params 映射 |
| `megatron/core/dist_checkpointing/dict_utils.py` | `merge()` 添加空列表合并处理 |
| `megatron/core/dist_checkpointing/mapping.py` | `apply_factory_merges()` 跳过越界索引 |

## 数据流总结

### 保存流程
```
LayerWiseDistributedOptimizer
  → shard_params() 保存 global_param_groups
  → 每个 rank 14 个本地参数
  → sharded_state_dict() 生成 112 个 groups (14 ShardedTensor + 98 空列表)
  → checkpoint 系统合并所有 rank 数据
```

### 加载流程
```
LayerWiseDistributedOptimizer
  → _sync_global_float16_structure() 计算 global_float16_groups
  → sharded_state_dict(is_loading=True) 设置 global_float16_groups
  → Float16OptimizerWithFloat16Params 生成匹配结构
  → checkpoint 加载后 merge() 返回 14 个 tensor
  → load_state_dict() 检测到本地映射场景，直接 flatten 并映射
```

## 测试

运行 `muon_script.sh`:
1. 第一次运行：训练 50 步后保存 checkpoint
2. 第二次运行：加载 checkpoint 继续训练到 100 步

成功标准：iteration 51-52 的 loss 和 grad_norm 保持平稳（无 spike）
