---
status: Accepted
applies_to:
  - src/local_decision_model/to_mlx_chat.py
  - src/local_decision_model/evaluate.py
  - src/local_decision_model/build_dataset.py
---

# ADR-0002: Unified JSON schema with baked-in system prompt

## Status

Accepted

## Date

2026-10-04

## Context

Multiple source datasets (MultiNLI, BoolQ, Banking77, AG News, SST-5) plus
synthetic generators must feed one training format, and the model must obey a
strict output contract at inference. Divergence between the training prompt
and the serving prompt would silently degrade the contract.

## Decision

1. All sources normalize to one record schema:
   `{"state", "question", "options": [{label, key, description}], "answer", "source"}`.
2. A single `SYSTEM_PROMPT` constant is **baked into every training example**
   (`to_mlx_chat.py`) and reproduced verbatim in `evaluate.py`.

## Consequences

- Adding a source = writing one generator function returning unified records.
- Prompt drift is a known failure mode; the duplicate constant is documented in
  AGENTS.md and should be tested for equality (see `tests/test_pipeline.py`).
