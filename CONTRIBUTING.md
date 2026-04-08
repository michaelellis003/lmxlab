# Contributing

## Setup

```bash
git clone https://github.com/michaelellis003/lmxlab.git
cd lmxlab
uv sync --extra dev
uv run pre-commit install
uv run pre-commit install --hook-type commit-msg
```

## Workflow

1. Branch from `main` (`feat/`, `fix/`, `docs/`, `refactor/`, `test/`)
2. Write tests in `tests/`, run with `uv run pytest`
3. Check lint: `uv run ruff check src/ tests/ recipes/`
4. Open a PR — CI runs lint, docs build, and tests on macOS

## Commit messages

`type: description` format, enforced by pre-commit:

```
feat: add sliding window attention
fix: correct RoPE dimension calculation
```

## Code style

- 79-char lines, double quotes, Google-style docstrings
- Type annotations on public functions
- Ruff handles formatting: `ruff format src/ tests/ recipes/`

## Testing

Use behavioral tests for ML code: shape checks, determinism with fixed seeds,
loss-decreases-after-training, no NaN/Inf. Keep tests fast with tiny configs.

## Adding architectures

1. Add a config factory in `src/lmxlab/models/`
2. Register new components in the appropriate registry
3. Add a `_tiny()` config and tests
4. Update CLI and `models/__init__.py`
