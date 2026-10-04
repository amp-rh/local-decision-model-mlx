# Design Intent

What this system is for, and the constraints that make it correct.

## Purpose

Train a small, cheap, fully-local **decision classifier** (Jev-style): given a
`state`, a `question`, and labeled `options`, return exactly one option label
as JSON. It replaces LLM calls in routing/gating/rating decisions where the
judgment is narrow, high-volume, and latency-sensitive.

## Preconditions

- P1. **Dataset available before training.** `make data` must have produced
  `data/train.jsonl` and `data/dev.jsonl` before `make chat-format` or
  `make train` runs.
- P2. **Unified schema holds for every record:**
  `{"state", "question", "options": [{label, key, description}], "answer", "source"}`
  — `answer` must be the `key` of exactly one option.
- P3. **Apple Silicon + MLX + LM Studio/vLLM-compatible serving** for the
  train/serve steps (`scripts/04_train.sh`, `scripts/05_fuse_serve.sh`).
- P4. **Same system prompt at train and inference time** (baked into the chat
  JSONL in `to_mlx_chat.py`, reproduced in `evaluate.py`).

## Invariants

- I1. `state` is data, never instructions. The system prompt says so; the
  schema keeps `state` a single string that is never concatenated with
  instruction text outside the fixed template.
- I2. Output contract: `{"label": "<letter>", "key": "<key>"}` and nothing
  else. `temperature=0`, `max_tokens=8` enforce it at serving time.
- I3. Determinism of data: every generator/adapter is seeded (`seed=42`,
  `random.Random(42)`), so `make data` is reproducible.
- I4. Option labels are single letters `A`–`H` (`LETTERS`); the answer letter
  is derivable from `answer` + `options` alone.
- I5. Train/dev are disjoint and drawn from the same shuffled pool (dev =
  max(500, 2%)); evaluation (`make eval`) only ever reads dev.

## Non-goals

- No cloud, no API calls at train or serve time ($0 pipeline).
- No explanation output — this is a classifier, not a chat assistant.
- No multi-turn or tool use.

## Acceptable trade-offs

- LoRA (rank 32) over full fine-tuning: quality headroom traded for a 48GB Mac
  fitting the job.
- Lenient eval fallback (`raw.startswith(expected)`) counts bare-letter
  outputs as correct but tracks malformed rate separately.
