# Interface Contracts

Key interfaces, their invariants, and where they're defined.
Updated by `/design` (new interfaces) and `/critique` (violations).

---

## INT-001: Attention Contract

**Defined in:** `src/lmxlab/core/attention.py` (AttentionBase)
**Implementations:** MHA, GQA, SlidingWindowGQA, MLA, GatedDeltaNet

**Signature:**
```python
def __call__(
    self,
    x: mx.array,        # (batch, seq_len, d_model)
    mask: mx.array | None,
    cache: Any | None,
) -> tuple[mx.array, Any]:  # (output, new_cache)
```

**Invariants:**
- Output shape == input shape: `(batch, seq_len, d_model)`
- Cache is opaque to callers — each impl manages its own format
- `mask=None` means no masking (full attention)
- Constructor takes `BlockConfig` as sole required argument

---

## INT-002: Tokenizer Protocol

**Defined in:** `src/lmxlab/data/tokenizer.py`
**Implementations:** CharTokenizer, TiktokenTokenizer, HFTokenizer

**Signature:**
```python
class Tokenizer(Protocol):
    @property
    def vocab_size(self) -> int: ...
    def encode(self, text: str) -> list[int]: ...
    def decode(self, tokens: list[int]) -> str: ...
```

**Invariants:**
- `decode(encode(text))` should approximate `text` (may not be exact
  for subword tokenizers)
- `vocab_size` is constant after construction
- `encode` returns valid token IDs in `[0, vocab_size)`

---

## INT-003: Callback Protocol

**Defined in:** `src/lmxlab/training/callbacks.py`
**Implementations:** MetricsLogger, ThroughputMonitor, EarlyStopping

**Signature:**
```python
class Callback(Protocol):
    def on_train_begin(self, config: TrainConfig) -> None: ...
    def on_train_end(self, history: list[dict[str, Any]]) -> None: ...
    def on_step_end(self, step: int, metrics: dict[str, Any]) -> None: ...
    def on_eval_end(self, step: int, metrics: dict[str, Any]) -> None: ...
```

**Invariants:**
- All methods are optional (Protocol with default no-ops)
- Callbacks must not modify the model or optimizer
- `on_step_end` is called every step; `on_eval_end` only at eval intervals
- `metrics` dict always contains `'loss'` key

---

## INT-004: FFN Contract

**Defined in:** `src/lmxlab/core/ffn.py` (FFNBase)
**Implementations:** StandardFFN, GatedFFN, MoEFFN, SharedExpertMoEFFN

**Signature:**
```python
def __call__(self, x: mx.array) -> mx.array:
    # x: (batch, seq_len, d_model) -> (batch, seq_len, d_model)
```

**Invariants:**
- Output shape == input shape
- Constructor takes `BlockConfig` (MoE reads extra fields)
- No side effects (pure function of input)

---

## INT-005: ConfigurableBlock Assembly

**Defined in:** `src/lmxlab/core/block.py`
**Consumers:** LanguageModel

**Contract:** ConfigurableBlock resolves components from registries
at construction time using string keys in `BlockConfig`:
- `config.attention` → `attention_registry`
- `config.ffn` → `ffn_registry`
- `config.norm` → `norm_registry`
- `config.position` → `position_registry`

**Invariants:**
- All registry keys must be valid at construction time
- Components are instantiated once and reused for all forward passes
- Pre-norm vs post-norm controlled by `config.pre_norm`

---

## INT-006: ExperimentLog Contract

**Defined in:** `src/lmxlab/experiments/tracking.py`

**Invariants:**
- Append-only: `log()` only adds, never modifies
- `load()` returns entries in chronological order
- `best()` only considers entries with `status='keep'`
- File format: one JSON object per line (JSONL)

---

## INT-007: μP Optimizer Contract

**Defined in:** `src/lmxlab/training/optimizers.py`
(planned — DES-004)

**Signature:**
```python
def create_mup_optimizer(
    config: TrainConfig,
    width_mult: float,
) -> optim.MultiOptimizer:
```

**Invariants:**
- Returns `MultiOptimizer` with 3 groups: embed, hidden, head
- Embed group uses base LR (no scaling)
- Hidden/head groups use LR / width_mult
- When `width_mult == 1.0`, equivalent to `create_optimizer()`
- Filter functions match on parameter path substrings:
  `'embed'` for embeddings, `'head'` for output, rest is hidden
- Each group uses the same schedule type from `config`
