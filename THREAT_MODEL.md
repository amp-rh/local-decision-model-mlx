# Threat Model

Scope: a fully-local ML pipeline (dataset build → LoRA train → local inference
endpoint). No cloud services, no external APIs at train/serve time.

## Assets

- A1. Trained adapter weights (`adapters/`, `model/`) — the artifact.
- A2. Local inference endpoint (`localhost:8080`) — OpenAI-compatible server.
- A3. Datasets (`data/`) — derived from public HF datasets + synthetic text.
- A4. Developer machine / Mac Mini.

## Trust boundaries

- TB1. Network → local machine (dataset downloads via HF `datasets`).
- TB2. Inference clients → localhost:8080 server.
- TB3. Untrusted text → model input (`state` field).

## Threats and mitigations

| # | Threat | Vector | Mitigation |
|---|--------|--------|------------|
| T1 | Prompt injection via `state` | Untrusted text in the state field instructs the model | System prompt declares state as data-only; output constrained to one option label (`max_tokens=8`, `temperature=0`); consumers must parse JSON and validate the label against options |
| T2 | Malicious/poisoned HF dataset | Compromised dataset repo downloaded during `make data` | Pin well-known datasets by canonical names; treat downloaded text as untrusted data (T1 mitigations apply); run on a dev box, not a production machine |
| T3 | Endpoint exposure | Server bound beyond localhost | Serve scripts bind to localhost:8080 only; do not add auth-free exposure to LAN — verify before changing `05_fuse_serve.sh` |
| T4 | Supply chain (pip deps) | Malicious package in dev/data extras | Minimal pinned dependency set (`pyproject.toml`); CI runs `bandit` + ruff security rules; review new deps before adding |
| T5 | Secrets leakage | Keys committed by accident | No secrets needed anywhere in the pipeline ($0/local); `detect-private-key` pre-commit hook; `.gitignore` excludes data/ and model artifacts |

## Residual risks (accepted)

- A local attacker on the machine can read weights/data — out of scope (single-
  user dev machine).
- `state` injection can still flip a borderline decision; downstream consumers
  must treat outputs as advisory judgments, not authenticated facts.
