#!/usr/bin/env bash
set -e

# 1. 进入脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 2. 下载数据集
wget -O benchmark.zip "https://raw.githubusercontent.com/richardodliu/Megatron_benchmark/main/benchmark.zip"

# 3. 解压 zip
unzip -o benchmark.zip

# 4. 解压成功后删除 zip
rm -f benchmark.zip