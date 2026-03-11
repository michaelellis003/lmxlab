# Contributing to lmt-metal

Thanks for your interest in contributing! lmt-metal is an educational project,
so clarity and simplicity are valued over performance optimization.

## Setup

```bash
git clone https://github.com/michaelellis003/lmt-metal.git
cd lmt-metal
pip install -e ".[dev]"
```

## Development workflow

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feat/my-feature
   ```

2. Write tests first (TDD). Tests go in `tests/`:
   ```bash
   pytest tests/test_my_module.py -v
   ```

3. Implement the feature in `src/lmt_metal/`.

4. Verify everything passes:
   ```bash
   pytest                          # All tests
   ruff check src/ tests/          # Lint
   ruff format --check src/ tests/ # Formatting
   ```

5. Open a PR against `main`.

## Branch naming

- `feat/` — new features
- `fix/` — bug fixes
- `docs/` — documentation changes
- `refactor/` — code restructuring
- `test/` — test additions

## Commit messages

Follow the `type: description` format:

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
ruff check --fix src/ tests/   # Auto-fix lint issues
ruff format src/ tests/         # Auto-format
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
- **Educational clarity.** Comments should explain *why*, not just *what*.

## Adding a new architecture

1. Create `src/lmt_metal/models/myarch.py` with a config factory function
2. Register any new components in the appropriate registry
3. Add a `myarch_tiny()` config for tests
4. Add tests in `tests/test_architectures.py`
5. Update `src/lmt_metal/models/__init__.py` exports
6. Add to CLI in `src/lmt_metal/cli.py`
7. Document in `docs/models/index.md`
