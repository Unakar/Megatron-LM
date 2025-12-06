# SECTION 1: Global Configuration (common.pt)

Status:   Loaded Successfully
Step:     50

[Model Architecture]
  - num_layers                    : 28
  - hidden_size                   : 1024
  - num_attention_heads           : 16
  - seq_length                    : 4096
  - tensor_model_parallel_size    : 1
  - pipeline_model_parallel_size  : 1

# SECTION 2: Categorized Parameter Inventory

Overview:
  - Model Weights:       11 entries
  - Muon States:         4 entries (Large Matrices)
  - AdamW States:        14 entries (Vectors/Norms)
  - FP32 Master Weights: 11 entries

## 1. Model Weights (BF16/FP16 - For Inference)

Key Name / Pattern                                                | Count    | Full Shape           | Note      
decoder.final_layernorm.weight                                    | 1        | [1024]               | bfloat16 (TP)
decoder.layers.mlp.linear_fc1.layer_norm_weight                   | 1        | [28, 1024]           | bfloat16 (TP)
decoder.layers.mlp.linear_fc1.weight                              | 1        | [28, 6144, 1024]     | bfloat16 (TP)
decoder.layers.mlp.linear_fc2.weight                              | 1        | [28, 1024, 3072]     | bfloat16 (TP)
decoder.layers.self_attention.k_layernorm.weight                  | 1        | [28, 128]            | bfloat16 (TP)
decoder.layers.self_attention.linear_proj.weight                  | 1        | [28, 1024, 2048]     | bfloat16 (TP)
decoder.layers.self_attention.linear_qkv.layer_norm_weight        | 1        | [28, 1024]           | bfloat16 (TP)
decoder.layers.self_attention.linear_qkv.weight                   | 1        | [28, 4096, 1024]     | bfloat16 (TP)
decoder.layers.self_attention.q_layernorm.weight                  | 1        | [28, 128]            | bfloat16 (TP)
embedding.word_embeddings.weight                                  | 1        | [151680, 1024]       | bfloat16 (TP)
output_layer.weight                                               | 1        | [151680, 1024]       | bfloat16 (TP)

## 2. Muon Optimizer States (Momentum Buffer - For Linear Layers)

Key Name / Pattern                                                | Count    | Full Shape           | Note      
optimizer.state.momentum_buffer.decoder.layers.mlp.linear_fc1.weight | 1        | [28, 6144, 1024]     | float32 (TP)
optimizer.state.momentum_buffer.decoder.layers.mlp.linear_fc2.weight | 1        | [28, 1024, 3072]     | float32 (TP)
optimizer.state.momentum_buffer.decoder.layers.self_attention.linear_proj.weight | 1        | [28, 1024, 2048]     | float32 (TP)
optimizer.state.momentum_buffer.decoder.layers.self_attention.linear_qkv.weight | 1        | [28, 4096, 1024]     | float32 (TP)

## 3. AdamW Optimizer States (Exp Avg/Sq - For Norms/Embeddings)

Key Name / Pattern                                                | Count    | Full Shape           | Note      
optimizer.state.exp_avg.decoder.final_layernorm.weight            | 1        | [1024]               | float32 (TP)
optimizer.state.exp_avg.decoder.layers.mlp.linear_fc1.layer_norm_weight | 1        | [28, 1024]           | float32 (TP)
optimizer.state.exp_avg.decoder.layers.self_attention.k_layernorm.weight | 1        | [28, 128]            | float32 (TP)
optimizer.state.exp_avg.decoder.layers.self_attention.linear_qkv.layer_norm_weight | 1        | [28, 1024]           | float32 (TP)
optimizer.state.exp_avg.decoder.layers.self_attention.q_layernorm.weight | 1        | [28, 128]            | float32 (TP)
optimizer.state.exp_avg.embedding.word_embeddings.weight          | 1        | [151680, 1024]       | float32 (TP)
optimizer.state.exp_avg.output_layer.weight                       | 1        | [151680, 1024]       | float32 (TP)
optimizer.state.exp_avg_sq.decoder.final_layernorm.weight         | 1        | [1024]               | float32 (TP)
optimizer.state.exp_avg_sq.decoder.layers.mlp.linear_fc1.layer_norm_weight | 1        | [28, 1024]           | float32 (TP)
optimizer.state.exp_avg_sq.decoder.layers.self_attention.k_layernorm.weight | 1        | [28, 128]            | float32 (TP)
optimizer.state.exp_avg_sq.decoder.layers.self_attention.linear_qkv.layer_norm_weight | 1        | [28, 1024]           | float32 (TP)
optimizer.state.exp_avg_sq.decoder.layers.self_attention.q_layernorm.weight | 1        | [28, 128]            | float32 (TP)
optimizer.state.exp_avg_sq.embedding.word_embeddings.weight       | 1        | [151680, 1024]       | float32 (TP)
optimizer.state.exp_avg_sq.output_layer.weight                    | 1        | [151680, 1024]       | float32 (TP)

## 4. FP32 Master Weights (For Training Precision)

Key Name / Pattern                                                | Count    | Full Shape           | Note      
optimizer.state.fp32_param.decoder.final_layernorm.weight         | 1        | [1024]               | float32 (TP)
optimizer.state.fp32_param.decoder.layers.mlp.linear_fc1.layer_norm_weight | 1        | [28, 1024]           | float32 (TP)
optimizer.state.fp32_param.decoder.layers.mlp.linear_fc1.weight   | 1        | [28, 6144, 1024]     | float32 (TP)
optimizer.state.fp32_param.decoder.layers.mlp.linear_fc2.weight   | 1        | [28, 1024, 3072]     | float32 (TP)
optimizer.state.fp32_param.decoder.layers.self_attention.k_layernorm.weight | 1        | [28, 128]            | float32 (TP)
optimizer.state.fp32_param.decoder.layers.self_attention.linear_proj.weight | 1        | [28, 1024, 2048]     | float32 (TP)
optimizer.state.fp32_param.decoder.layers.self_attention.linear_qkv.layer_norm_weight | 1        | [28, 1024]           | float32 (TP)
optimizer.state.fp32_param.decoder.layers.self_attention.linear_qkv.weight | 1        | [28, 4096, 1024]     | float32 (TP)
optimizer.state.fp32_param.decoder.layers.self_attention.q_layernorm.weight | 1        | [28, 128]            | float32 (TP)
optimizer.state.fp32_param.embedding.word_embeddings.weight       | 1        | [151680, 1024]       | float32 (TP)
optimizer.state.fp32_param.output_layer.weight                    | 1        | [151680, 1024]       | float32 (TP)

## 5. Miscellaneous (RNG & Internal States)

Key Name / Pattern                                                | Count    | Full Shape           | Note      
decoder.final_layernorm._extra_state/shard_0_1                    | 1        | (Binary)             | Meta      
......     
decoder.layers.self_attention.q_layernorm._extra_state/shard_9_28 | 1        | (Binary)             | Meta      
rng_state/shard_0.0_1.{N}                                         | x1       | (Binary)             | Meta      

# SECTION 3: Physical Storage Detail

## Logical Shard ID: 0

Physical File(s): __0_0.distcp, __0_1.distcp
Total Size:       2002.03 MB

[Content Inventory]

Param Name                                                             | Slice Shape          | Offset         
decoder.final_layernorm.weight                                         | [1024]               | [0]            
decoder.layers.mlp.linear_fc1.layer_norm_weight                        | [1, 1024]            | [0, 0]         
decoder.layers.mlp.linear_fc1.weight                                   | [1, 3072, 1024]      | [0, 0, 0]      
decoder.layers.mlp.linear_fc2.weight                                   | [1, 1024, 3072]      | [0, 0, 0]      
decoder.layers.self_attention.k_layernorm.weight                       | [1, 128]             | [0, 0]         
decoder.layers.self_attention.linear_proj.weight                       | [1, 1024, 2048]      | [0, 0, 0]      
decoder.layers.self_attention.linear_qkv.layer_norm_weight             | [1, 1024]            | [0, 0]         
decoder.layers.self_attention.linear_qkv.weight                        | [1, 4096, 1024]      | [16, 0, 0]     
decoder.layers.self_attention.q_layernorm.weight                       | [1, 128]             | [0, 0]         
embedding.word_embeddings.weight                                       | [151680, 1024]       | [0, 0]         
optimizer.state.exp_avg.decoder.final_layernorm.weight                 | [1024]               | [0]            
optimizer.state.exp_avg.decoder.layers.mlp.linear_fc1.layer_norm_weight | [1, 1024]            | [1, 0]         
optimizer.state.exp_avg.decoder.layers.self_attention.k_layernorm.weight | [1, 128]             | [0, 0]         
optimizer.state.exp_avg.decoder.layers.self_attention.linear_qkv.layer_norm_weight | [1, 1024]            | [0, 0]         
optimizer.state.exp_avg.decoder.layers.self_attention.q_layernorm.weight | [1, 128]             | [0, 0]         
optimizer.state.exp_avg.embedding.word_embeddings.weight               | [151680, 1024]       | [0, 0]         
optimizer.state.exp_avg.output_layer.weight                            | [151680, 1024]       | [0, 0]         
optimizer.state.exp_avg_sq.decoder.final_layernorm.weight              | [1024]               | [0]            
optimizer.state.exp_avg_sq.decoder.layers.mlp.linear_fc1.layer_norm_weight | [1, 1024]            | [1, 0]         
optimizer.state.exp_avg_sq.decoder.layers.self_attention.k_layernorm.weight | [1, 128]             | [0, 0]         
optimizer.state.exp_avg_sq.decoder.layers.self_attention.linear_qkv.layer_norm_weight | [1, 1024]            | [0, 0]         
optimizer.state.exp_avg_sq.decoder.layers.self_attention.q_layernorm.weight | [1, 128]             | [0, 0]         
optimizer.state.exp_avg_sq.embedding.word_embeddings.weight            | [151680, 1024]       | [0, 0]         
optimizer.state.exp_avg_sq.output_layer.weight                         | [151680, 1024]       | [0, 0]         
optimizer.state.fp32_param.decoder.final_layernorm.weight              | [1024]               | [0]            
optimizer.state.fp32_param.decoder.layers.mlp.linear_fc1.layer_norm_weight | [1, 1024]            | [1, 0]         
optimizer.state.fp32_param.decoder.layers.mlp.linear_fc1.weight        | [1, 3072, 1024]      | [0, 0, 0]      
optimizer.state.fp32_param.decoder.layers.mlp.linear_fc2.weight        | [1, 1024, 3072]      | [0, 0, 0]      
optimizer.state.fp32_param.decoder.layers.self_attention.k_layernorm.weight | [1, 128]             | [0, 0]         
optimizer.state.fp32_param.decoder.layers.self_attention.linear_proj.weight | [1, 1024, 2048]      | [0, 0, 0]      
optimizer.state.fp32_param.decoder.layers.self_attention.linear_qkv.layer_norm_weight | [1, 1024]            | [0, 0]         
optimizer.state.fp32_param.decoder.layers.self_attention.linear_qkv.weight | [1, 4096, 1024]      | [0, 0, 0]      
optimizer.state.fp32_param.decoder.layers.self_attention.q_layernorm.weight | [1, 128]             | [0, 0]         
optimizer.state.fp32_param.embedding.word_embeddings.weight            | [151680, 1024]       | [0, 0]         
optimizer.state.fp32_param.output_layer.weight                         | [151680, 1024]       | [0, 0]         
optimizer.state.momentum_buffer.decoder.layers.mlp.linear_fc1.weight   | [1, 3072, 1024]      | [0, 0, 0]      
optimizer.state.momentum_buffer.decoder.layers.mlp.linear_fc2.weight   | [1, 1024, 3072]      | [0, 0, 0]      
optimizer.state.momentum_buffer.decoder.layers.self_attention.linear_proj.weight | [1, 1024, 2048]      | [0, 0, 0]      
optimizer.state.momentum_buffer.decoder.layers.self_attention.linear_qkv.weight | [1, 4096, 1024]      | [0, 0, 0]      
output_layer.weight                                                    | [151680, 1024]       | [0, 0]         


Total items: 40

## Logical Shard ID: 1

Physical File(s): __1_0.distcp, __1_1.distcp
Total Size:       646.34 MB

[Content Inventory]

Param Name                                                             | Slice Shape          | Offset         

decoder.layers.mlp.linear_fc1.layer_norm_weight                        | [1, 1024]            | [1, 0]         
decoder.layers.mlp.linear_fc1.weight                                   | [1, 3072, 1024]      | [0, 3072, 0]   
decoder.layers.mlp.linear_fc2.weight                                   | [1, 1024, 3072]      | [4, 0, 0]      
decoder.layers.self_attention.k_layernorm.weight                       | [1, 128]             | [1, 0]         
decoder.layers.self_attention.linear_proj.weight                       | [1, 1024, 2048]      | [3, 0, 0]      
decoder.layers.self_attention.linear_qkv.layer_norm_weight             | [1, 1024]            | [1, 0]         
decoder.layers.self_attention.linear_qkv.weight                        | [1, 4096, 1024]      | [19, 0, 0]     
decoder.layers.self_attention.q_layernorm.weight                       | [1, 128]             | [1, 0]         
optimizer.state.exp_avg.decoder.layers.mlp.linear_fc1.layer_norm_weight | [1, 1024]            | [3, 0]         
optimizer.state.exp_avg.decoder.layers.self_attention.k_layernorm.weight | [1, 128]             | [2, 0]         
optimizer.state.exp_avg.decoder.layers.self_attention.linear_qkv.layer_norm_weight | [1, 1024]            | [2, 0]         
optimizer.state.exp_avg.decoder.layers.self_attention.q_layernorm.weight | [1, 128]             | [2, 0]         
optimizer.state.exp_avg_sq.decoder.layers.mlp.linear_fc1.layer_norm_weight | [1, 1024]            | [3, 0]         
optimizer.state.exp_avg_sq.decoder.layers.self_attention.k_layernorm.weight | [1, 128]             | [2, 0]         
optimizer.state.exp_avg_sq.decoder.layers.self_attention.linear_qkv.layer_norm_weight | [1, 1024]            | [2, 0]         
optimizer.state.exp_avg_sq.decoder.layers.self_attention.q_layernorm.weight | [1, 128]             | [2, 0]         
optimizer.state.fp32_param.decoder.layers.mlp.linear_fc1.layer_norm_weight | [1, 1024]            | [3, 0]         
optimizer.state.fp32_param.decoder.layers.mlp.linear_fc1.weight        | [1, 3072, 1024]      | [0, 3072, 0]   
optimizer.state.fp32_param.decoder.layers.mlp.linear_fc2.weight        | [1, 1024, 3072]      | [2, 0, 0]      
optimizer.state.fp32_param.decoder.layers.self_attention.k_layernorm.weight | [1, 128]             | [2, 0]         
optimizer.state.fp32_param.decoder.layers.self_attention.linear_proj.weight | [1, 1024, 2048]      | [2, 0, 0]      
optimizer.state.fp32_param.decoder.layers.self_attention.linear_qkv.layer_norm_weight | [1, 1024]            | [2, 0]         
optimizer.state.fp32_param.decoder.layers.self_attention.linear_qkv.weight | [1, 4096, 1024]      | [2, 0, 0]      
optimizer.state.fp32_param.decoder.layers.self_attention.q_layernorm.weight | [1, 128]             | [2, 0]         
optimizer.state.momentum_buffer.decoder.layers.mlp.linear_fc1.weight   | [1, 3072, 1024]      | [0, 3072, 0]   
optimizer.state.momentum_buffer.decoder.layers.mlp.linear_fc2.weight   | [1, 1024, 3072]      | [2, 0, 0]      
optimizer.state.momentum_buffer.decoder.layers.self_attention.linear_proj.weight | [1, 1024, 2048]      | [2, 0, 0]      
optimizer.state.momentum_buffer.decoder.layers.self_attention.linear_qkv.weight | [1, 4096, 1024]      | [2, 0, 0]      

Total items: 28

## Logical Shard ID: 2

Physical File(s): __2_0.distcp, __2_1.distcp
Total Size:       2449.67 MB

[Content Inventory]

Param Name                                                             | Slice Shape          | Offset         
decoder.layers.mlp.linear_fc1.layer_norm_weight                        | [1, 1024]            | [2, 0]         
decoder.layers.mlp.linear_fc1.weight                                   | [1, 3072, 1024]      | [2, 0, 0]      
decoder.layers.mlp.linear_fc2.weight                                   | [1, 1024, 3072]      | [8, 0, 0]      
decoder.layers.self_attention.k_layernorm.weight                       | [1, 128]             | [2, 0]         
decoder.layers.self_attention.linear_proj.weight                       | [1, 1024, 2048]      | [7, 0, 0]      
decoder.layers.self_attention.linear_qkv.layer_norm_weight             | [1, 1024]            | [2, 0]         
decoder.layers.self_attention.linear_qkv.weight                        | [1, 4096, 1024]      | [22, 0, 0]     
decoder.layers.self_attention.q_layernorm.weight                       | [1, 128]             | [2, 0]         
optimizer.state.exp_avg.decoder.layers.mlp.linear_fc1.layer_norm_weight | [1, 1024]            | [5, 0]         
optimizer.state.exp_avg.decoder.layers.self_attention.k_layernorm.weight | [1, 128]             | [4, 0]         
optimizer.state.exp_avg.decoder.layers.self_attention.linear_qkv.layer_norm_weight | [1, 1024]            | [4, 0]         
optimizer.state.exp_avg.decoder.layers.self_attention.q_layernorm.weight | [1, 128]             | [4, 0]         
optimizer.state.exp_avg_sq.decoder.layers.mlp.linear_fc1.layer_norm_weight | [1, 1024]            | [5, 0]         
optimizer.state.exp_avg_sq.decoder.layers.self_attention.k_layernorm.weight | [1, 128]             | [4, 0]         
optimizer.state.exp_avg_sq.decoder.layers.self_attention.linear_qkv.layer_norm_weight | [1, 1024]            | [4, 0]         
optimizer.state.exp_avg_sq.decoder.layers.self_attention.q_layernorm.weight | [1, 128]             | [4, 0]         
optimizer.state.fp32_param.decoder.layers.mlp.linear_fc1.layer_norm_weight | [1, 1024]            | [5, 0]         
optimizer.state.fp32_param.decoder.layers.mlp.linear_fc1.weight        | [1, 3072, 1024]      | [2, 0, 0]      
optimizer.state.fp32_param.decoder.layers.mlp.linear_fc2.weight        | [1, 1024, 3072]      | [4, 0, 0]      
optimizer.state.fp32_param.decoder.layers.self_attention.k_layernorm.weight | [1, 128]             | [4, 0]         
optimizer.state.fp32_param.decoder.layers.self_attention.linear_proj.weight | [1, 1024, 2048]      | [4, 0, 0]      
optimizer.state.fp32_param.decoder.layers.self_attention.linear_qkv.layer_norm_weight | [1, 1024]            | [4, 0]         
optimizer.state.fp32_param.decoder.layers.self_attention.linear_qkv.weight | [1, 4096, 1024]      | [4, 0, 0]      
optimizer.state.fp32_param.decoder.layers.self_attention.q_layernorm.weight | [1, 128]             | [4, 0]         
optimizer.state.momentum_buffer.decoder.layers.mlp.linear_fc1.weight   | [1, 3072, 1024]      | [2, 0, 0]      
optimizer.state.momentum_buffer.decoder.layers.mlp.linear_fc2.weight   | [1, 1024, 3072]      | [4, 0, 0]      
optimizer.state.momentum_buffer.decoder.layers.self_attention.linear_proj.weight | [1, 1024, 2048]      | [4, 0, 0]      
optimizer.state.momentum_buffer.decoder.layers.self_attention.linear_qkv.weight | [1, 4096, 1024]      | [4, 0, 0]      

Total items: 28

## Logical Shard ID: 3

Physical File(s): __3_0.distcp, __3_1.distcp
Total Size:       644.38 MB

[Content Inventory]

Param Name                                                             | Slice Shape          | Offset         
decoder.layers.mlp.linear_fc1.layer_norm_weight                        | [1, 1024]            | [3, 0]         
decoder.layers.mlp.linear_fc1.weight                                   | [1, 3072, 1024]      | [3, 3072, 0]   
decoder.layers.mlp.linear_fc2.weight                                   | [1, 1024, 3072]      | [12, 0, 0]     
decoder.layers.self_attention.k_layernorm.weight                       | [1, 128]             | [3, 0]         
decoder.layers.self_attention.linear_proj.weight                       | [1, 1024, 2048]      | [11, 0, 0]     
decoder.layers.self_attention.linear_qkv.layer_norm_weight             | [1, 1024]            | [3, 0]         
decoder.layers.self_attention.linear_qkv.weight                        | [1, 4096, 1024]      | [25, 0, 0]     
decoder.layers.self_attention.q_layernorm.weight                       | [1, 128]             | [3, 0]         
optimizer.state.exp_avg.decoder.layers.mlp.linear_fc1.layer_norm_weight | [1, 1024]            | [7, 0]         
optimizer.state.exp_avg.decoder.layers.self_attention.k_layernorm.weight | [1, 128]             | [6, 0]         
optimizer.state.exp_avg.decoder.layers.self_attention.linear_qkv.layer_norm_weight | [1, 1024]            | [6, 0]         
optimizer.state.exp_avg.decoder.layers.self_attention.q_layernorm.weight | [1, 128]             | [6, 0]         
optimizer.state.exp_avg_sq.decoder.layers.mlp.linear_fc1.layer_norm_weight | [1, 1024]            | [7, 0]         
optimizer.state.exp_avg_sq.decoder.layers.self_attention.k_layernorm.weight | [1, 128]             | [6, 0]         
optimizer.state.exp_avg_sq.decoder.layers.self_attention.linear_qkv.layer_norm_weight | [1, 1024]            | [6, 0]         
optimizer.state.exp_avg_sq.decoder.layers.self_attention.q_layernorm.weight | [1, 128]             | [6, 0]         
optimizer.state.fp32_param.decoder.layers.mlp.linear_fc1.layer_norm_weight | [1, 1024]            | [7, 0]         
optimizer.state.fp32_param.decoder.layers.mlp.linear_fc1.weight        | [1, 3072, 1024]      | [2, 3072, 0]   
optimizer.state.fp32_param.decoder.layers.mlp.linear_fc2.weight        | [1, 1024, 3072]      | [6, 0, 0]      
optimizer.state.fp32_param.decoder.layers.self_attention.k_layernorm.weight | [1, 128]             | [6, 0]         
optimizer.state.fp32_param.decoder.layers.self_attention.linear_proj.weight | [1, 1024, 2048]      | [6, 0, 0]      
optimizer.state.fp32_param.decoder.layers.self_attention.linear_qkv.layer_norm_weight | [1, 1024]            | [6, 0]         
optimizer.state.fp32_param.decoder.layers.self_attention.linear_qkv.weight | [1, 4096, 1024]      | [6, 0, 0]      
optimizer.state.fp32_param.decoder.layers.self_attention.q_layernorm.weight | [1, 128]             | [6, 0]         
optimizer.state.momentum_buffer.decoder.layers.mlp.linear_fc1.weight   | [1, 3072, 1024]      | [2, 3072, 0]   
optimizer.state.momentum_buffer.decoder.layers.mlp.linear_fc2.weight   | [1, 1024, 3072]      | [6, 0, 0]      
optimizer.state.momentum_buffer.decoder.layers.self_attention.linear_proj.weight | [1, 1024, 2048]      | [6, 0, 0]      
optimizer.state.momentum_buffer.decoder.layers.self_attention.linear_qkv.weight | [1, 4096, 1024]      | [6, 0, 0]      

Total items: 28


## Logical Shard ID: 4

Physical File(s): __4_0.distcp, __4_1.distcp
Total Size:       646.56 MB

[Content Inventory]

Param Name                                                             | Slice Shape          | Offset         
decoder.layers.mlp.linear_fc1.layer_norm_weight                        | [1, 1024]            | [4, 0]         
decoder.layers.mlp.linear_fc1.weight                                   | [1, 3072, 1024]      | [5, 0, 0]      
decoder.layers.mlp.linear_fc2.weight                                   | [1, 1024, 3072]      | [16, 0, 0]     
decoder.layers.self_attention.k_layernorm.weight                       | [1, 128]             | [4, 0]         
decoder.layers.self_attention.linear_proj.weight                       | [1, 1024, 2048]      | [16, 0, 0]     
decoder.layers.self_attention.linear_qkv.layer_norm_weight             | [1, 1024]            | [4, 0]         
decoder.layers.self_attention.linear_qkv.weight                        | [1, 4096, 1024]      | [15, 0, 0]     
decoder.layers.self_attention.q_layernorm.weight                       | [1, 128]             | [4, 0]         
optimizer.state.exp_avg.decoder.layers.mlp.linear_fc1.layer_norm_weight | [1, 1024]            | [9, 0]         
optimizer.state.exp_avg.decoder.layers.self_attention.k_layernorm.weight | [1, 128]             | [8, 0]         
optimizer.state.exp_avg.decoder.layers.self_attention.linear_qkv.layer_norm_weight | [1, 1024]            | [8, 0]         
optimizer.state.exp_avg.decoder.layers.self_attention.q_layernorm.weight | [1, 128]             | [8, 0]         
optimizer.state.exp_avg_sq.decoder.layers.mlp.linear_fc1.layer_norm_weight | [1, 1024]            | [9, 0]         
optimizer.state.exp_avg_sq.decoder.layers.self_attention.k_layernorm.weight | [1, 128]             | [8, 0]         
optimizer.state.exp_avg_sq.decoder.layers.self_attention.linear_qkv.layer_norm_weight | [1, 1024]            | [8, 0]         
optimizer.state.exp_avg_sq.decoder.layers.self_attention.q_layernorm.weight | [1, 128]             | [8, 0]         
optimizer.state.fp32_param.decoder.layers.mlp.linear_fc1.layer_norm_weight | [1, 1024]            | [9, 0]         
optimizer.state.fp32_param.decoder.layers.mlp.linear_fc1.weight        | [1, 3072, 1024]      | [4, 0, 0]      
optimizer.state.fp32_param.decoder.layers.mlp.linear_fc2.weight        | [1, 1024, 3072]      | [8, 0, 0]      
optimizer.state.fp32_param.decoder.layers.self_attention.k_layernorm.weight | [1, 128]             | [8, 0]         
optimizer.state.fp32_param.decoder.layers.self_attention.linear_proj.weight | [1, 1024, 2048]      | [8, 0, 0]      
optimizer.state.fp32_param.decoder.layers.self_attention.linear_qkv.layer_norm_weight | [1, 1024]            | [8, 0]         
optimizer.state.fp32_param.decoder.layers.self_attention.linear_qkv.weight | [1, 4096, 1024]      | [8, 0, 0]      
optimizer.state.fp32_param.decoder.layers.self_attention.q_layernorm.weight | [1, 128]             | [8, 0]         
optimizer.state.momentum_buffer.decoder.layers.mlp.linear_fc1.weight   | [1, 3072, 1024]      | [4, 0, 0]      
optimizer.state.momentum_buffer.decoder.layers.mlp.linear_fc2.weight   | [1, 1024, 3072]      | [8, 0, 0]      
optimizer.state.momentum_buffer.decoder.layers.self_attention.linear_proj.weight | [1, 1024, 2048]      | [8, 0, 0]      
optimizer.state.momentum_buffer.decoder.layers.self_attention.linear_qkv.weight | [1, 4096, 1024]      | [8, 0, 0]      

Total items: 28

## Logical Shard ID: 5

Physical File(s): __5_0.distcp, __5_1.distcp
Total Size:       644.32 MB

[Content Inventory]

Param Name                                                             | Slice Shape          | Offset         
decoder.layers.mlp.linear_fc1.layer_norm_weight                        | [1, 1024]            | [5, 0]         
decoder.layers.mlp.linear_fc1.weight                                   | [1, 3072, 1024]      | [6, 3072, 0]   
decoder.layers.mlp.linear_fc2.weight                                   | [1, 1024, 3072]      | [20, 0, 0]     
decoder.layers.self_attention.k_layernorm.weight                       | [1, 128]             | [5, 0]         
decoder.layers.self_attention.linear_proj.weight                       | [1, 1024, 2048]      | [21, 0, 0]     
decoder.layers.self_attention.linear_qkv.layer_norm_weight             | [1, 1024]            | [5, 0]         
decoder.layers.self_attention.linear_qkv.weight                        | [1, 4096, 1024]      | [18, 0, 0]     
decoder.layers.self_attention.q_layernorm.weight                       | [1, 128]             | [5, 0]         
optimizer.state.exp_avg.decoder.layers.mlp.linear_fc1.layer_norm_weight | [1, 1024]            | [11, 0]        
optimizer.state.exp_avg.decoder.layers.self_attention.k_layernorm.weight | [1, 128]             | [10, 0]        
optimizer.state.exp_avg.decoder.layers.self_attention.linear_qkv.layer_norm_weight | [1, 1024]            | [10, 0]        
optimizer.state.exp_avg.decoder.layers.self_attention.q_layernorm.weight | [1, 128]             | [10, 0]        
optimizer.state.exp_avg_sq.decoder.layers.mlp.linear_fc1.layer_norm_weight | [1, 1024]            | [11, 0]        
optimizer.state.exp_avg_sq.decoder.layers.self_attention.k_layernorm.weight | [1, 128]             | [10, 0]        
optimizer.state.exp_avg_sq.decoder.layers.self_attention.linear_qkv.layer_norm_weight | [1, 1024]            | [10, 0]        
optimizer.state.exp_avg_sq.decoder.layers.self_attention.q_layernorm.weight | [1, 128]             | [10, 0]        
optimizer.state.fp32_param.decoder.layers.mlp.linear_fc1.layer_norm_weight | [1, 1024]            | [11, 0]        
optimizer.state.fp32_param.decoder.layers.mlp.linear_fc1.weight        | [1, 3072, 1024]      | [4, 3072, 0]   
optimizer.state.fp32_param.decoder.layers.mlp.linear_fc2.weight        | [1, 1024, 3072]      | [10, 0, 0]     
optimizer.state.fp32_param.decoder.layers.self_attention.k_layernorm.weight | [1, 128]             | [10, 0]        
optimizer.state.fp32_param.decoder.layers.self_attention.linear_proj.weight | [1, 1024, 2048]      | [10, 0, 0]     
optimizer.state.fp32_param.decoder.layers.self_attention.linear_qkv.layer_norm_weight | [1, 1024]            | [10, 0]        
optimizer.state.fp32_param.decoder.layers.self_attention.linear_qkv.weight | [1, 4096, 1024]      | [10, 0, 0]     
optimizer.state.fp32_param.decoder.layers.self_attention.q_layernorm.weight | [1, 128]             | [10, 0]        
optimizer.state.momentum_buffer.decoder.layers.mlp.linear_fc1.weight   | [1, 3072, 1024]      | [4, 3072, 0]   
optimizer.state.momentum_buffer.decoder.layers.mlp.linear_fc2.weight   | [1, 1024, 3072]      | [10, 0, 0]     
optimizer.state.momentum_buffer.decoder.layers.self_attention.linear_proj.weight | [1, 1024, 2048]      | [10, 0, 0]     
optimizer.state.momentum_buffer.decoder.layers.self_attention.linear_qkv.weight | [1, 4096, 1024]      | [10, 0, 0]     

Total items: 28


## Logical Shard ID: 6

Physical File(s): __6_0.distcp, __6_1.distcp
Total Size:       672.17 MB

[Content Inventory]

Param Name                                                             | Slice Shape          | Offset
decoder.layers.mlp.linear_fc1.layer_norm_weight                        | [1, 1024]            | [6, 0]         
decoder.layers.mlp.linear_fc1.weight                                   | [1, 3072, 1024]      | [8, 0, 0]      
decoder.layers.mlp.linear_fc2.weight                                   | [1, 1024, 3072]      | [24, 0, 0]     
decoder.layers.self_attention.k_layernorm.weight                       | [1, 128]             | [6, 0]         
decoder.layers.self_attention.linear_proj.weight                       | [1, 1024, 2048]      | [26, 0, 0]     
decoder.layers.self_attention.linear_qkv.layer_norm_weight             | [1, 1024]            | [6, 0]         
decoder.layers.self_attention.linear_qkv.weight                        | [1, 4096, 1024]      | [21, 0, 0]     
decoder.layers.self_attention.q_layernorm.weight                       | [1, 128]             | [6, 0]         
optimizer.state.exp_avg.decoder.layers.mlp.linear_fc1.layer_norm_weight | [1, 1024]            | [13, 0]        
optimizer.state.exp_avg.decoder.layers.self_attention.k_layernorm.weight | [1, 128]             | [12, 0]        
optimizer.state.exp_avg.decoder.layers.self_attention.linear_qkv.layer_norm_weight | [1, 1024]            | [12, 0]        
optimizer.state.exp_avg.decoder.layers.self_attention.q_layernorm.weight | [1, 128]             | [12, 0]        
optimizer.state.exp_avg_sq.decoder.layers.mlp.linear_fc1.layer_norm_weight | [1, 1024]            | [13, 0]        
optimizer.state.exp_avg_sq.decoder.layers.self_attention.k_layernorm.weight | [1, 128]             | [12, 0]        
optimizer.state.exp_avg_sq.decoder.layers.self_attention.linear_qkv.layer_norm_weight | [1, 1024]            | [12, 0]        
optimizer.state.exp_avg_sq.decoder.layers.self_attention.q_layernorm.weight | [1, 128]             | [12, 0]        
optimizer.state.fp32_param.decoder.layers.mlp.linear_fc1.layer_norm_weight | [1, 1024]            | [13, 0]        
optimizer.state.fp32_param.decoder.layers.mlp.linear_fc1.weight        | [1, 3072, 1024]      | [6, 0, 0]      
optimizer.state.fp32_param.decoder.layers.mlp.linear_fc2.weight        | [1, 1024, 3072]      | [12, 0, 0]     
optimizer.state.fp32_param.decoder.layers.self_attention.k_layernorm.weight | [1, 128]             | [12, 0]        
optimizer.state.fp32_param.decoder.layers.self_attention.linear_proj.weight | [1, 1024, 2048]      | [12, 0, 0]     
optimizer.state.fp32_param.decoder.layers.self_attention.linear_qkv.layer_norm_weight | [1, 1024]            | [12, 0]        
optimizer.state.fp32_param.decoder.layers.self_attention.linear_qkv.weight | [1, 4096, 1024]      | [12, 0, 0]     
optimizer.state.fp32_param.decoder.layers.self_attention.q_layernorm.weight | [1, 128]             | [12, 0]        
optimizer.state.momentum_buffer.decoder.layers.mlp.linear_fc1.weight   | [1, 3072, 1024]      | [6, 0, 0]      
optimizer.state.momentum_buffer.decoder.layers.mlp.linear_fc2.weight   | [1, 1024, 3072]      | [12, 0, 0]     
optimizer.state.momentum_buffer.decoder.layers.self_attention.linear_proj.weight | [1, 1024, 2048]      | [12, 0, 0]     
optimizer.state.momentum_buffer.decoder.layers.self_attention.linear_qkv.weight | [1, 4096, 1024]      | [12, 0, 0]     

Total items: 28


## Logical Shard ID: 7

Physical File(s): __7_0.distcp, __7_1.distcp
Total Size:       644.49 MB

[Content Inventory]

Param Name                                                             | Slice Shape          | Offset         
decoder.layers.mlp.linear_fc1.layer_norm_weight                        | [1, 1024]            | [7, 0]         
decoder.layers.mlp.linear_fc1.weight                                   | [1, 3072, 1024]      | [10, 0, 0]     
decoder.layers.mlp.linear_fc2.weight                                   | [1, 1024, 3072]      | [3, 0, 0]      
decoder.layers.self_attention.k_layernorm.weight                       | [1, 128]             | [7, 0]         
decoder.layers.self_attention.linear_proj.weight                       | [1, 1024, 2048]      | [15, 0, 0]     
decoder.layers.self_attention.linear_qkv.layer_norm_weight             | [1, 1024]            | [7, 0]         
decoder.layers.self_attention.linear_qkv.weight                        | [1, 4096, 1024]      | [24, 0, 0]     
decoder.layers.self_attention.q_layernorm.weight                       | [1, 128]             | [7, 0]         
optimizer.state.exp_avg.decoder.layers.mlp.linear_fc1.layer_norm_weight | [1, 1024]            | [15, 0]        
optimizer.state.exp_avg.decoder.layers.self_attention.k_layernorm.weight | [1, 128]             | [14, 0]        
optimizer.state.exp_avg.decoder.layers.self_attention.linear_qkv.layer_norm_weight | [1, 1024]            | [14, 0]        
optimizer.state.exp_avg.decoder.layers.self_attention.q_layernorm.weight | [1, 128]             | [14, 0]        
optimizer.state.exp_avg_sq.decoder.layers.mlp.linear_fc1.layer_norm_weight | [1, 1024]            | [15, 0]        
optimizer.state.exp_avg_sq.decoder.layers.self_attention.k_layernorm.weight | [1, 128]             | [14, 0]        
optimizer.state.exp_avg_sq.decoder.layers.self_attention.linear_qkv.layer_norm_weight | [1, 1024]            | [14, 0]        
optimizer.state.exp_avg_sq.decoder.layers.self_attention.q_layernorm.weight | [1, 128]             | [14, 0]        
optimizer.state.fp32_param.decoder.layers.mlp.linear_fc1.layer_norm_weight | [1, 1024]            | [15, 0]        
optimizer.state.fp32_param.decoder.layers.mlp.linear_fc1.weight        | [1, 3072, 1024]      | [6, 3072, 0]   
optimizer.state.fp32_param.decoder.layers.mlp.linear_fc2.weight        | [1, 1024, 3072]      | [14, 0, 0]     
optimizer.state.fp32_param.decoder.layers.self_attention.k_layernorm.weight | [1, 128]             | [14, 0]        
optimizer.state.fp32_param.decoder.layers.self_attention.linear_proj.weight | [1, 1024, 2048]      | [14, 0, 0]     
optimizer.state.fp32_param.decoder.layers.self_attention.linear_qkv.layer_norm_weight | [1, 1024]            | [14, 0]        
optimizer.state.fp32_param.decoder.layers.self_attention.linear_qkv.weight | [1, 4096, 1024]      | [14, 0, 0]     
optimizer.state.fp32_param.decoder.layers.self_attention.q_layernorm.weight | [1, 128]             | [14, 0]        
optimizer.state.momentum_buffer.decoder.layers.mlp.linear_fc1.weight   | [1, 3072, 1024]      | [6, 3072, 0]   
optimizer.state.momentum_buffer.decoder.layers.mlp.linear_fc2.weight   | [1, 1024, 3072]      | [14, 0, 0]     
optimizer.state.momentum_buffer.decoder.layers.self_attention.linear_proj.weight | [1, 1024, 2048]      | [14, 0, 0]     
optimizer.state.momentum_buffer.decoder.layers.self_attention.linear_qkv.weight | [1, 4096, 1024]      | [14, 0, 0]     

Total items: 28