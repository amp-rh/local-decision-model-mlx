#!/usr/bin/env bash
# Fuse the trained adapter into the weights and serve a local OpenAI-compatible endpoint.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-mlx-community/Qwen3-4B-Instruct-2507-4bit}"
PORT="${PORT:-8080}"

mlx_lm.fuse --model "$MODEL" --adapter-path adapters --save-path model/my-jev-4b
mlx_lm.server --model model/my-jev-4b --port "$PORT"
