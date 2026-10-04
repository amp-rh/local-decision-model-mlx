.PHONY: data data-hf chat-format train serve eval clean

# Build the unified dataset. Requires `pip install datasets` (network access);
# run once on any machine, then copy data/ to the Mac training box.
data:
	python3 scripts/02_build_dataset.py

# Convert unified JSONL to MLX chat format (cheap, CPU-only)
chat-format:
	python3 scripts/03_to_mlx_chat.py data data/mlx

# LoRA fine-tune on the Mac (Apple Silicon, MLX)
train:
	bash scripts/04_train.sh

# Fuse adapter + serve localhost endpoint
serve:
	bash scripts/05_fuse_serve.sh

# Evaluate against the held-out dev split (server must be running)
eval:
	python3 scripts/06_eval.py

clean:
	rm -rf data adapters model
