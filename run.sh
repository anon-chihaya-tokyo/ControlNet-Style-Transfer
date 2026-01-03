#!/bin/bash
echo "🚀 正在启动 AI 风格迁移系统 (工程版)..."

# 激活环境
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate comfyui 
fi

# 启动入口文件
HF_ENDPOINT=https://hf-mirror.com python main.py
