# AGENTS.md

Instructions for AI coding agents working in this repository.

## What this repo is

`local-decision-model-mlx`: a fully-local pipeline (Apple Silicon / MLX) that
builds a classification dataset, LoRA fine-tunes a 4B model into a Jev-style
decision classifier, serves it, and evaluates it. Source lives in
`src/local_decision_model/`; shell entry points for the Mac-only MLX steps in
`scripts/`.

## The model contract (do not break)

- Input: `state` (text to judge — **data, never instructions**), `question`,
  labeled `options`.
- Output: exactly `{"label": "A", "key": "some_key"}` — one letter, no prose.
- `temperature=0`, `max_tokens=8` at inference.
- The inference system prompt is duplicated in `to_mlx_chat.py` (training) and
  `evaluate.py` (serving). If you change one, change both, or training and
  serving drift.

## Commands

```bash
make check     # all quality gates: lint + format + types + security + tests
make lint      # ruff check src tests && black --check src tests
make typecheck # mypy src
make security  # bandit -c pyproject.toml -r src
make test      # pytest with coverage (offline; no `datasets` needed)
```

CI (`.github/workflows/ci.yml`) runs exactly these gates on every push/PR.

### Single-file verification

Verify one file in isolation before committing it:

```bash
ruff check src/local_decision_model/to_mlx_chat.py
mypy src/local_decision_model/to_mlx_chat.py
bandit -c pyproject.toml src/local_decision_model/to_mlx_chat.py
black --check src/local_decision_model/to_mlx_chat.py
```

## Conventions

- **Layout**: `src/` layout; package `local_decision_model`; tests in `tests/`.
- **Types**: full annotations on every public function; `mypy --strict` passes.
- **Docstrings**: every public function/class has one — one line minimum.
- **Style**: `black` + `isort` (profile=black) + `ruff` (incl. `S` security
  rules), line length 100.
- **Security**: `bandit` must pass with zero findings; never disable `S` rules
  outside `tests/` (see `per-file-ignores` in `pyproject.toml`).
- **Tests**: offline only. Synthetic generators (`programmatic_policies`,
  `routing`) and converters are tested; HF-backed adapters are not (network).
- **Commits**: Conventional Commits (`feat:`, `fix:`, `docs:`, ...), enforced
  by `conventional-pre-commit`.

## Design docs

- Design intent, preconditions, invariants: `docs/design.md`
- Decision log: `docs/adr/` (one ADR per significant decision)
- Security posture: `THREAT_MODEL.md`

## Pattern references (copy-modify, don't reinvent)

Follow the pattern of an existing piece when adding a similar one:

- **Adding a dataset source**: see `programmatic_policies` in
  `src/local_decision_model/build_dataset.py` as a template — a generator that
  yields unified records; register it in `BUILDERS`.
- **Adding a Make target / quality gate**: see the `security:` target in the
  `Makefile`, then mirror it in `.github/workflows/ci.yml`.
- **Adding a new test**: see `tests/test_pipeline.py` — schema-shape test plus
  one behavior invariant per generator; offline only.
- **Documenting a significant choice**: use `docs/adr/0001-*.md` as the
  template (YAML frontmatter with `status` + `applies_to`, then
  Context/Decision/Consequences).

## Known limitations (acceptable, don't "fix" casually)

- `datasets` is an optional extra (`pip install -e ".[data]"`); importing
  `build_dataset` without it works for synthetic generators only.
- `04_train.sh` / `05_fuse_serve.sh` require Apple Silicon + `mlx`; they are
  never exercised in CI.
