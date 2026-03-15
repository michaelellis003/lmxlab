# Pattern Catalog

Design patterns used in the codebase, where they appear, and why.
Based on the Pattern Language concept (Alexander) — document the
recurring solutions so new code can reuse them consistently.

---

## PAT-001: Factory Functions (Config Factories)

**Category:** Creational
**Used in:** `src/lmxlab/models/*.py`
**Principle:** Separate object configuration from object construction.

Each architecture (GPT, LLaMA, Gemma, DeepSeek, etc.) is a factory
function returning `ModelConfig`, not a separate model class.

```python
config = llama_config(d_model=512, n_heads=8, n_kv_heads=4)
model = LanguageModel(config)  # same class for all architectures
```

**When to use:** Any time multiple configurations share the same
underlying structure. Prefer over class hierarchies.
**Related:** DES-001

---

## PAT-002: Typed Registry

**Category:** Structural
**Used in:** `src/lmxlab/core/registry.py`, applied to attention,
FFN, norm, position registries.
**Principle:** Decouple component implementation from assembly.

```python
@attention_registry.register('gqa')
class GQA(AttentionBase): ...

# Later, in ConfigurableBlock:
attn_cls = attention_registry.get(config.attention)
```

**When to use:** When components are selected by name at runtime
and new components should be addable without modifying consumers.
**Related:** DES-002

---

## PAT-003: Protocol (Structural Typing)

**Category:** Behavioral
**Used in:** `Tokenizer` (data/tokenizer.py), `Callback`
(training/callbacks.py)
**Principle:** Define contracts without forcing inheritance.

```python
class Tokenizer(Protocol):
    @property
    def vocab_size(self) -> int: ...
    def encode(self, text: str) -> list[int]: ...
    def decode(self, tokens: list[int]) -> str: ...
```

**When to use:** When multiple implementations exist (possibly from
different libraries) and you want duck typing with static checking.
**Related:** DES-003, INT-002

---

## PAT-004: Frozen Dataclass Configuration

**Category:** Structural
**Used in:** `BlockConfig`, `ModelConfig` (core/config.py),
`TrainConfig` (training/config.py)
**Principle:** Immutable configuration prevents accidental mutation.

```python
@dataclass(frozen=True)
class BlockConfig:
    attention: str = 'mha'
    d_model: int = 512
    ...
```

**When to use:** All configuration objects. Frozen dataclasses are
hashable, safe to share, and prevent subtle mutation bugs.

---

## PAT-005: Callback Protocol (Observer)

**Category:** Behavioral
**Used in:** `training/callbacks.py`, `training/trainer.py`
**Principle:** Decouple training loop from monitoring/logging.

4 lifecycle hooks: `on_train_begin`, `on_step_end`, `on_eval_end`,
`on_train_end`. Trainer calls hooks; callbacks observe without
coupling to trainer internals.

**When to use:** Any loop that needs pluggable monitoring,
early stopping, or checkpointing.

---

## PAT-006: Append-Only Log (Event Sourcing Lite)

**Category:** Behavioral
**Used in:** `ExperimentLog` (experiments/tracking.py)
**Principle:** Immutable, append-only logs are the source of truth.

```python
log.log(LogEntry(experiment='run-1', val_bpb=1.42))
entries = log.load()  # reconstruct state from log
```

**When to use:** Experiment tracking, audit trails, any data where
history matters more than current state.

---

## PAT-007: Lazy Imports

**Category:** Performance
**Used in:** HuggingFace integration (`models/convert.py`,
`data/tokenizer.py` HFTokenizer)
**Principle:** Don't pay for what you don't use.

Heavy dependencies (`transformers`, `datasets`) are imported inside
`__init__` methods, not at module level. Base library stays light.

**When to use:** Optional dependencies that not all users need.
