# Security Policy

## Reporting a vulnerability

Open a private security advisory via GitHub ("Security" → "Advisories") rather
than a public issue. This project has no production deployment and no secrets,
but weight artifacts and the local inference server are in scope.

## Supported versions

Only `main` is supported.

## Tooling

- `bandit` SAST on every commit (pre-commit + CI)
- `ruff` security rules (`S`) on every commit
- `pip-audit` dependency scanning in CI (weekly Dependabot updates)
- Threat analysis: see [THREAT_MODEL.md](THREAT_MODEL.md)
