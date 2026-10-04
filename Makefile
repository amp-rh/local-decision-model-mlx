.PHONY: data data-hf chat-format train serve eval check lint format typecheck test security clean

# src/ layout: make the package importable without installing
export PYTHONPATH := src

# Build the unified dataset. Requires `pip install datasets` (network access);
# run once on any machine, then copy data/ to the Mac training box.
data:
	python3 -m local_decision_model.build_dataset

# Convert unified JSONL to MLX chat format (cheap, CPU-only)
chat-format:
	python3 -m local_decision_model.to_mlx_chat data data/mlx

# LoRA fine-tune on the Mac (Apple Silicon, MLX)
train:
	bash scripts/04_train.sh

# Fuse adapter + serve localhost endpoint
serve:
	bash scripts/05_fuse_serve.sh

# Evaluate against the held-out dev split (server must be running)
eval:
	python3 -m local_decision_model.evaluate

# Run every quality gate: lint, format check, types, security, tests
check: lint typecheck security test

lint:
	ruff check src tests
	black --check src tests

format:
	ruff check --fix src tests
	black src tests

typecheck:
	mypy src

security:
	bandit -c pyproject.toml -r src

test:
	python3 -m pytest --cov=local_decision_model --cov-report=term-missing

clean:
	rm -rf data adapters model .pytest_cache .ruff_cache .mypy_cache .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
