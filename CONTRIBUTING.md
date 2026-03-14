# Contributing to lmxlab

Thanks for your interest in contributing! lmxlab is a research platform,
so clarity and rapid iteration are valued over production optimization.

## Setup

```bash
git clone https://github.com/michaelellis003/lmxlab.git
cd lmxlab
uv sync --extra dev
uv run pre-commit install
uv run pre-commit install --hook-type commit-msg
```

The pre-commit hooks will automatically:
- Run ruff lint and format checks on staged files
- Verify `uv.lock` stays in sync with `pyproject.toml`
- Enforce conventional commit message format

## Development workflow

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feat/my-feature
   ```

2. Write tests first (TDD). Tests go in `tests/`:
   ```bash
   uv run pytest tests/test_my_module.py -v
   ```

3. Implement the feature in `src/lmxlab/`.

4. Verify everything passes locally before pushing:
   ```bash
   uv run pytest                                    # All tests
   uv run ruff check src/ tests/ recipes/            # Lint
   uv run ruff format --check src/ tests/ recipes/   # Formatting
   uv run mkdocs build --strict                      # Docs build
   ```

5. Open a PR against `main`. CI must pass before merging.

## CI pipeline

Every PR runs three jobs:

- **lint** (ubuntu): ruff check + format on `src/`, `tests/`, `recipes/`
- **docs** (ubuntu): `mkdocs build --strict` catches broken links/refs
- **test** (macos-14): pytest on Apple Silicon (MLX requires M-series)

All three must pass before merging. Do not bypass CI with `--admin`.

## Keeping `uv.lock` in sync

If you change `pyproject.toml` (add/remove/update dependencies), you must
regenerate the lockfile:

```bash
uv lock
```

The `uv-lock` pre-commit hook catches this automatically. If CI fails with
"lockfile needs to be updated", run `uv lock` and commit the result.

## Branch naming

- `feat/` — new features
- `fix/` — bug fixes
- `docs/` — documentation changes
- `refactor/` — code restructuring
- `test/` — test additions

## Commit messages

Follow the `type: description` format (enforced by pre-commit hook):

```
feat: add sliding window attention
fix: correct RoPE dimension calculation
docs: expand installation guide
test: add CLI command tests
refactor: simplify MoE routing logic
```

## Code style

- **Line length:** 79 characters
- **Quotes:** double quotes (enforced by ruff)
- **Docstrings:** Google style
- **Type annotations:** required on all public functions
- **Imports:** sorted by ruff (stdlib, third-party, local)

Ruff handles formatting and linting:
```bash
ruff check --fix src/ tests/ recipes/   # Auto-fix lint issues
ruff format src/ tests/ recipes/         # Auto-format
```

## Testing

- Use **behavioral tests** for ML code:
  - Shape tests: output dimensions are correct
  - Invariance tests: same input + seed = same output
  - Directional tests: loss decreases after training
  - Minimum functionality: no NaN, no Inf
- Keep tests fast: use tiny model configs (`gpt_tiny()`, `llama_tiny()`)
- Mark slow tests with `@pytest.mark.slow`

## Architecture guidelines

- **Config factories, not subclasses.** New architectures should be
  config factory functions, not new model classes.
- **Registry pattern.** New attention/FFN/norm types should register
  themselves in the appropriate registry.
- **Simplicity bias.** When two approaches achieve similar results,
  prefer the simpler one.
- **Clarity.** Comments should explain *why*, not just *what*.

## Citations and attribution

Every new building block (attention, FFN, position encoding, SSM,
etc.) must cite its source paper with an arXiv ID in the module
docstring:

```python
"""My new attention variant.

Reference: Author (Year, arXiv:XXXX.XXXXX)
"""
```

Code adapted from reference implementations must note the source:

```python
# Cross-references:
# - org/repo filename.py (canonical implementation)
# - HuggingFace transformers modeling_xxx.py
```

Use `Reference:` for the originating paper and `Cross-references:`
for implementation sources consulted during development.

## Adding a new architecture

1. Create `src/lmxlab/models/myarch.py` with a config factory function
2. Register any new components in the appropriate registry
3. Add a `myarch_tiny()` config for tests
4. Add tests in `tests/test_architectures.py`
5. Update `src/lmxlab/models/__init__.py` exports
6. Add to CLI in `src/lmxlab/cli.py`
7. Document in `docs/models/index.md`
