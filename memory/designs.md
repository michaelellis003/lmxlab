# Design Registry

Pre-implementation design documents. Each design is created via
`/design` before implementation and reviewed via `/critique` after.

## Status Key

| Status | Meaning |
|--------|---------|
| `draft` | In progress, not yet reviewed |
| `approved` | Reviewed, ready to implement |
| `implemented` | Implementation complete |
| `superseded` | Replaced by a newer design |
| `abandoned` | Decided not to implement |

---

## DES-001: Config Factories Over Class Hierarchies

**Status:** implemented
**Date:** pre-project (migrated from devlog)
**Author:** project founder

**Problem:** Need to support 8+ transformer architectures without
code duplication or artificial class boundaries.

**Constraints:**
- Architectures differ in configuration, not structure
- Must be easy to add new architectures
- Must support hybrid configurations (mix features from different archs)

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| A. Class hierarchy (`LlamaModel(BaseModel)`) | Familiar OOP, easy to grep | Artificial boundaries, code duplication |
| B. Config factories + single model class | No duplication, easy to mix features | Harder to grep for individual architectures |
| C. Builder pattern | Flexible composition | Verbose, over-engineered for this use case |

**Decision:** Option B — config factories.
**Trade-off:** Greppability sacrificed for composability. Mitigated
by well-documented factory functions in `models/*.py`.
**Related:** DEC-001 (methodology), PAT-001 (factory pattern)

---

## DES-002: Registry Pattern for Components

**Status:** implemented
**Date:** pre-project (migrated from devlog)
**Author:** project founder

**Problem:** Need to add new component types (attention, FFN, norm,
position) without modifying existing assembly code.

**Constraints:**
- `ConfigurableBlock` must resolve components at construction time
- New components must work without touching existing code
- MoE FFNs need extra constructor arguments

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| A. Registry pattern (string → factory) | Zero-touch extensibility | Indirection, harder to trace |
| B. Direct imports in block.py | Simple, explicit | Every new component modifies block.py |
| C. Plugin system with entry points | Most decoupled | Over-engineered for a single package |

**Decision:** Option A — typed `Registry[T]` with decorator registration.
**Trade-off:** Indirection vs extensibility. Registry makes adding
components trivial but makes code navigation harder.
**Related:** PAT-002 (registry pattern), INT-001 (attention contract)

---

## DES-003: Protocol-Based Contracts

**Status:** implemented
**Date:** pre-project (migrated from devlog)
**Author:** project founder

**Problem:** Components need shared contracts without forcing
inheritance from a common base class.

**Constraints:**
- Python's structural typing (Protocols) preferred over nominal typing
- Components from different libraries should interoperate
- Must be statically checkable (mypy)

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| A. Protocol classes (structural typing) | Duck typing, no inheritance needed | Less obvious contracts |
| B. Abstract base classes | Explicit, enforced at instantiation | Forces inheritance hierarchy |
| C. No formal contracts (just conventions) | Simplest | No static checking, easy to break |

**Decision:** Option A — Protocols for `Tokenizer` and `Callback`.
ABCs used sparingly for `AttentionBase` and `FFNBase` where shared
init logic justifies inheritance.
**Trade-off:** Mixed approach (Protocols + light ABCs) is pragmatic
but requires clear documentation of which pattern to use when.
**Related:** INT-002 (Tokenizer), INT-003 (Callback), PAT-003 (protocol)

---

## DES-004: μP (Maximal Update Parameterization)

**Status:** implemented
**Date:** 2026-03-13
**Author:** Claude (design), user (decision DEC-009)

**Problem:** Architecture comparisons at 3M params are confounded
by architecture-dependent optimal learning rates. μP enables
hyperparameter transfer across model widths, making small-scale
experiments more predictive of large-scale behavior.

**Constraints:**
- PAT-004: Config objects are frozen dataclasses — new fields need
  backward-compatible defaults
- PAT-001: μP works through existing factory functions, not new
  model classes
- INT-005: ConfigurableBlock assembly unchanged — μP affects init,
  optimizer, and attention scaling only
- INT-001: Attention contract unchanged — scaling is an internal
  detail of each attention implementation
- Must be opt-in: `mup=False` by default preserves SP (Standard
  Parameterization) behavior
- Must work with both GPT and LLaMA architectures
- Must work with tied and untied embeddings
- MLX's `MultiOptimizer` provides native per-group LR support

### Background: What μP Changes

μP (Yang et al. 2022, LIT-024) defines three core changes
relative to a "base" model width:

| Component | Standard (SP) | μP |
|-----------|--------------|-----|
| Hidden weight init | σ = 1/√fan_in | σ = 1/√fan_in (same) |
| Hidden layer LR | η | η / m (m = width_mult) |
| Output logits | logits | logits / m |
| Attention scale | 1/√d_head | 1/d_head |
| Embedding LR | η | η (same) |

Where `m = d_model / base_d_model` (the width multiplier).

The key insight: with these scaling rules, a learning rate found
optimal at `base_d_model` remains optimal at any larger `d_model`.

### Alternatives

**Option A: Config + MultiOptimizer (Minimal Integration)**

Add `mup_base_width: int | None` to `ModelConfig`. When set:
1. Attention classes read the width multiplier from config and
   adjust scaling internally
2. `LanguageModel.__init__` wraps the output logit computation
   with a `1/m` scaling factor
3. A new `create_mup_optimizer()` function returns a
   `MultiOptimizer` with LR groups
4. Recipes call `create_mup_optimizer()` instead of
   `create_optimizer()` when μP is enabled

μP logic lives in 3 places: attention (scaling), base model
(logits), optimizer factory (LR groups). No new classes.

**Option B: MuModel Wrapper Class**

Create a `MuLanguageModel(LanguageModel)` subclass that:
1. Overrides `__init__` to rescale weights after construction
2. Overrides `__call__` to scale output logits
3. Includes a class method to create the optimizer

μP logic is centralized in one class but breaks PAT-001 (config
factories) by introducing a model subclass.

**Option C: Standalone μP Module**

Create `src/lmxlab/training/mup.py` that provides:
1. `apply_mup_init(model, base_width)` — post-hoc weight rescaling
2. `create_mup_optimizer(model, config, base_width)` — optimizer
3. `mup_attention_scale(d_head, width_mult)` — scale helper

μP is entirely external to the model. Attention scaling handled
by a post-construction hook. Clean separation but requires
attention modules to accept an external scale override.

**Comparison:**

| Criterion | A: Config+MultiOpt | B: MuModel | C: Standalone |
|-----------|-------------------|------------|---------------|
| Complexity | Low | Medium | Medium |
| Consistency w/ PAT-001 | Yes (factories) | No (subclass) | Yes |
| Consistency w/ INT-001 | Minimal change | No change | Requires hook |
| Testability | Good | Good | Best |
| Invasiveness | 4-5 files changed | 1 new file | 1 new file |
| Deletability | Remove config field | Remove subclass | Remove module |

### Step 4: Trade-off Analysis

**A vs B:** A trades slightly more distributed logic for
consistency with existing patterns. B centralizes μP but
introduces a model subclass that breaks PAT-001.

**A vs C:** A trades a slightly larger diff for simpler
integration — attention scaling is handled naturally where
it's computed. C requires either an external scale hook on
attention or post-construction weight modification.

**B vs C:** Both create new files, but B couples μP to the
model class while C keeps it external. C is more deletable.

### Step 5: Decision

**Option A: Config + MultiOptimizer.**

This is the simplest approach that respects existing patterns.
μP is opt-in via a single config field (`mup_base_width`).
The attention scale change is a 1-line modification where the
scale is already computed. The output logit scaling is a
1-line modification in `LanguageModel.__call__`. The optimizer
change uses MLX's native `MultiOptimizer`.

**What we're giving up:** Centralization. μP logic is spread
across config, attention, base model, and optimizer. But each
change is small (1-3 lines) and well-localized.

**Revisit when:** If we add more parameterization schemes
(e.g., depth-μP from LIT-025), a standalone module (Option C)
may become worthwhile.

### Step 6: Interface Contracts

**Modified: ModelConfig**
```python
@dataclass(frozen=True)
class ModelConfig:
    block: BlockConfig = field(default_factory=BlockConfig)
    vocab_size: int = 32000
    n_layers: int = 6
    tie_embeddings: bool = True
    block_configs: tuple[BlockConfig, ...] | None = None
    mup_base_width: int | None = None  # NEW
```

Invariants:
- `mup_base_width is None` → Standard Parameterization (SP)
- `mup_base_width is not None` → μP enabled
- `mup_base_width` must be <= `block.d_model`
- Width multiplier `m = block.d_model / mup_base_width`

**Modified: Attention scale (MHA, GQA, SlidingWindowGQA)**

Constructor accepts optional `mup_width_mult`:
```python
# In __init__:
if config_provides_mup:
    self.scale = self.head_dim ** -1.0  # 1/d
else:
    self.scale = self.head_dim ** -0.5  # 1/√d
```

The attention classes don't directly know about μP — they
read a flag from BlockConfig. A new field `mup: bool = False`
on BlockConfig controls this.

**New: BlockConfig.mup field**
```python
@dataclass(frozen=True)
class BlockConfig:
    ...
    mup: bool = False  # NEW — use μP attention scaling
```

**Modified: LanguageModel.__call__ (logit scaling)**
```python
# In __call__, after final_norm:
if self.config.mup_base_width is not None:
    m = (self.config.block.d_model
         / self.config.mup_base_width)
    logits = logits / m
```

**New: create_mup_optimizer()**
```python
def create_mup_optimizer(
    config: TrainConfig,
    width_mult: float,
) -> optim.MultiOptimizer:
    """Create optimizer with μP learning rate scaling.

    Args:
        config: Training configuration.
        width_mult: d_model / base_d_model ratio.

    Returns:
        MultiOptimizer with per-layer LR groups.
    """
```

Groups:
1. Embedding: base LR (no scaling)
2. Output head: base LR / m (when untied)
3. Hidden layers: base LR / m

**Modified: Factory functions**
```python
def gpt_config(
    ...,
    mup_base_width: int | None = None,
) -> ModelConfig:
```

When `mup_base_width` is set, factory also sets `mup=True`
on the BlockConfig.

### Step 7: Implementation Plan

1. **Add config fields** (config.py)
   - Add `mup: bool = False` to `BlockConfig`
   - Add `mup_base_width: int | None = None` to `ModelConfig`
   - Add `width_mult` property to `ModelConfig`

2. **Modify attention scaling** (attention.py)
   - In `MHA.__init__`: use `head_dim**-1.0` when `config.mup`
   - Same for `GQA.__init__` and `SlidingWindowGQA.__init__`

3. **Add output logit scaling** (base.py)
   - In `LanguageModel.__call__`: divide logits by width_mult
     when `mup_base_width` is set

4. **Create μP optimizer factory** (optimizers.py)
   - Add `create_mup_optimizer(config, width_mult)` function
   - Uses `MultiOptimizer` with embed/hidden/head groups
   - Filter functions match on parameter path names

5. **Update factory functions** (gpt.py, llama.py)
   - Add `mup_base_width` parameter
   - When set, pass `mup=True` to BlockConfig

6. **Write tests** (tests/test_mup.py)
   - Test attention scaling changes with mup=True
   - Test logit scaling with width_mult > 1
   - Test optimizer groups have correct LR ratios
   - Coordinate check: verify hidden activations don't grow
     with width (the standard μP validation)

7. **Update Trainer integration** (trainer.py)
   - Add μP-aware optimizer creation path
   - When `model.config.mup_base_width` is set, use
     `create_mup_optimizer` instead of `create_optimizer`

8. **Validate with experiment**
   - Train at base_width=64, find optimal LR
   - Train at target_width=256 with same LR
   - Verify loss curves match (coordinate check)

**Related:** DEC-009 (decision), LIT-024 (Yang et al. μP),
LIT-025 (Apple depth extension), PAT-001 (factories),
PAT-004 (frozen config), INT-001 (attention contract)
