#!/usr/bin/env bash
# LoRA fine-tune on Apple Silicon via MLX. Run on the Mac Mini.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-mlx-community/Qwen3-4B-Instruct-2507-4bit}"
ITERS="${ITERS:-4700}"
BATCH="${BATCH:-8}"
LR="${LR:-1e-4}"
RANK="${RANK:-32}"

mlx_lm.lora --train \
  --model "$MODEL" \
  --data data/mlx \
  --fine-tune-type lora \
  --num-layers "$RANK" \
  --batch-size "$BATCH" \
  --learning-rate "$LR" \
  --iters "$ITERS" \
  --steps-per-eval 200 \
  --adapter-path adapters
