# CI/CD Lessons

What works, what breaks, and what to watch out for in CI/CD.
Updated after build failures, flaky tests, and workflow changes.

## Pipeline Overview

Three CI jobs on every PR (all must pass):
- **lint** (ubuntu): `ruff check` + `ruff format --check`
- **docs** (ubuntu): `mkdocs build --strict`
- **test** (macos-14): `pytest -m "not slow"` on Apple Silicon

Release: GitHub Release → PyPI publish via OIDC trusted publishing.
Docs: auto-deploy to GitHub Pages on push to main.

---

## CI-001: Tests require macOS (Apple Silicon)

**Learned:** 2026-03-11 (from CI config)
**Impact:** high

MLX only runs on Apple Silicon. Tests run on `macos-14` runners,
not ubuntu. This means:
- Local testing requires an M-series Mac
- CI test failures can't be reproduced on Linux
- Tests must not assume CUDA or x86 behavior

---

## CI-002: Pre-commit hooks enforce conventional commits

**Learned:** 2026-03-11 (from .pre-commit-config.yaml)
**Impact:** medium

The `conventional-pre-commit` hook validates commit messages at
`commit-msg` stage. Allowed types: `feat`, `fix`, `refactor`,
`test`, `docs`, `chore`, `ci`, `build`, `perf`.

If a commit fails locally, check the message format first.
Install hooks with: `uv run pre-commit install --hook-type commit-msg`

---

## CI-003: uv.lock must stay in sync

**Learned:** 2026-03-11 (from CONTRIBUTING.md)
**Impact:** medium

If `pyproject.toml` changes, `uv.lock` must be regenerated with
`uv lock`. The `uv-lock` pre-commit hook catches this locally.
CI will fail with "lockfile needs to be updated" if missed.

---

## CI-004: Docs build is strict

**Learned:** 2026-03-11 (from ci.yml)
**Impact:** low

`mkdocs build --strict` catches broken links and missing
references. If you add a new module, make sure it's linked in
the docs nav (`mkdocs.yml`).

---

## CI-005: Large file check at 500KB

**Learned:** 2026-03-11 (from .pre-commit-config.yaml)
**Impact:** low

Pre-commit `check-added-large-files` is set to 500KB max.
Model weights, datasets, and checkpoints must go in `.gitignore`,
not the repo.

---

## CI-006: Run `uv lock` after version bumps in releases

**Learned:** 2026-03-15 (v0.3.0 release)
**Impact:** high

Bumping `version` in `pyproject.toml` without running `uv lock`
causes all three CI jobs to fail (`--locked` flag rejects stale
lockfiles). The `/release` skill bumps the version but does not
auto-run `uv lock`. **Always run `uv lock` after editing
pyproject.toml, before committing the release.**

v0.3.0 CI failed on the release commit; fixed in a follow-up push.

---

*Add new lessons after CI failures or workflow changes.*
