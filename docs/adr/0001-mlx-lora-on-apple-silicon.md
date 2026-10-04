---
status: Accepted
applies_to:
  - scripts/04_train.sh
  - Makefile
---

# ADR-0001: MLX LoRA on Apple Silicon instead of a rented cloud GPU

## Status

Accepted

## Date

2026-10-04

## Context

The baseline recipe (Together AI's "How to train your own Jev for $17")
fine-tunes on a rented H100 (~$17/run). Our goal is a $0, fully-local pipeline
runnable on a Mac Mini M4 (48GB); no data or spend leaves the machine.

## Decision

Use MLX (`mlx_lm.lora`, rank 32, ~4.7k iters ≈ 1 epoch) on Apple Silicon for
the 4B base model.

## Consequences

- $0 marginal cost per training run; overnight wall-clock on M4.
- Slightly lower ceiling than full fine-tuning on an H100 — accepted for this
  task domain (narrow classification).
- Mac-only constraint for `make train`/`make serve`; data prep stays
  cross-platform.
