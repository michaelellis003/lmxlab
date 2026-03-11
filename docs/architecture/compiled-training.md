# Compiled Training

This page explains how lmxlab's training loop uses `mx.compile` to fuse
the entire training step into a single optimized computation graph, and why
this matters for performance on Apple Silicon.

## The basic training step

Without compilation, a training step looks like this:

```python
# 1. Forward + backward (functional gradient)
loss, grads = loss_and_grad_fn(model, x, y)

# 2. Gradient clipping
grads, _ = optim.clip_grad_norm(grads, max_norm=1.0)

# 3. Optimizer update
optimizer.update(model, grads)

# 4. Force evaluation
mx.eval(loss, model.parameters(), optimizer.state)
```

Each of these steps builds a computation graph. MLX's lazy evaluation means
nothing actually runs until `mx.eval`. But without compilation, each call
to the step function creates a *new* graph every time — MLX must trace,
optimize, and schedule it from scratch.

## What `mx.compile` does

`mx.compile` traces the function once, captures the computation graph, and
reuses it on subsequent calls:

```python
# In Trainer.__init__:
if config.compile_step:
    self._step_fn = mx.compile(
        self._single_step,
        inputs=model.trainable_parameters(),
        outputs=model.trainable_parameters(),
    )
```

After the first call, the compiled function:

1. **Skips graph construction** — reuses the cached graph
2. **Enables kernel fusion** — combines multiple operations into single GPU kernels
3. **Optimizes memory layout** — plans buffer reuse across the graph
4. **Reduces Python overhead** — no Python-level tracing on subsequent calls

## The `inputs` and `outputs` contract

The `inputs` and `outputs` arguments tell the compiler which state is
read and mutated by the function:

```python
mx.compile(
    self._single_step,
    inputs=model.trainable_parameters(),   # State read by the function
    outputs=model.trainable_parameters(),  # State written by the function
)
```

This is necessary because `_single_step` mutates model parameters via
`optimizer.update`. Without declaring this, the compiler would not know
that the model's parameter arrays change between calls.

!!! warning "Getting inputs/outputs wrong"
    If you forget to include optimizer state in outputs, the optimizer's
    internal state (momentum, second moments for Adam) will not be
    updated correctly after the first step. In lmxlab, we pass
    `model.trainable_parameters()` which captures both the parameters
    and the optimizer's state through the model's parameter tree.

## When to compile (and when not to)

**Compile when:**

- Running production training loops (the default: `compile_step=True`)
- Profiling throughput — compilation gives realistic performance numbers
- The training step has no data-dependent control flow

**Don't compile when:**

- **Debugging** — compiled functions give less informative stack traces
- **Prototyping** — compilation adds startup latency for the first step
- **Variable-shape inputs** — if batch size or sequence length changes,
  the graph must be retraced (triggering recompilation)

Set `compile_step=False` in `TrainConfig` to disable:

```python
config = TrainConfig(compile_step=False)
```

## The compile gotcha: captured control flow

`mx.compile` traces the function once and caches the graph. Any
Python-level control flow that depends on *tensor values* is captured
at trace time and frozen:

```python
# BAD: condition depends on loss value (a tensor)
def step(x, y):
    loss, grads = loss_and_grad_fn(model, x, y)
    if loss > 10.0:  # This is evaluated ONCE at trace time!
        grads = scale_grads(grads, 0.1)
    optimizer.update(model, grads)
    return loss
```

After tracing, the `if` branch is permanently taken (or not), regardless
of the actual loss value. The compiled function becomes a fixed graph.

**The fix:** keep compiled functions free of data-dependent branches.
Use `mx.where` for conditional computation that should vary with data:

```python
# OK: mx.where is a graph operation, not Python control flow
scale = mx.where(loss > 10.0, 0.1, 1.0)
grads = tree_map(lambda g: g * scale, grads)
```

## Compilation and LoRA

LoRA fine-tuning works with compiled training. Since `apply_lora` freezes
all non-LoRA parameters, `model.trainable_parameters()` returns only
the LoRA matrices (lora_A and lora_B). The compiled step correctly
updates only these:

```python
apply_lora(model, rank=8)

# trainable_parameters() now returns only LoRA params
# The compiled step will only compute gradients for these
trainer = Trainer(model, TrainConfig(compile_step=True))
```

## Performance impact

The speedup from compilation depends on model size and step complexity.
Expected improvements on Apple Silicon (approximate, based on MLX
documentation and community benchmarks — actual results vary by
hardware, batch size, and sequence length):

| Scenario | Uncompiled | Compiled | Speedup |
|----------|-----------|----------|---------|
| Tiny model (64d, 2L) | ~1ms/step | ~0.8ms/step | ~1.3x |
| Small model (256d, 6L) | ~5ms/step | ~3ms/step | ~1.7x |
| Medium model (1024d, 12L) | ~30ms/step | ~15ms/step | ~2x |

The larger the model, the more opportunity for kernel fusion and the
greater the relative reduction in Python overhead. Use
`benchmark_compile.py` to measure on your specific hardware.

## How lmxlab structures the compiled step

The full compiled function in `Trainer._single_step`:

```python
def _single_step(self, x, y):
    # Forward pass + backward pass (functional)
    loss, grads = self._loss_and_grad(self.model, x, y)

    # Gradient clipping (functional — returns new grads)
    if self.config.max_grad_norm > 0:
        grads, _ = optim.clip_grad_norm(
            grads, max_norm=self.config.max_grad_norm
        )

    # Optimizer update (mutates model params in-place)
    self.optimizer.update(self.model, grads)
    return loss
```

Key design choices:

1. **Everything in one function** — forward, backward, clipping, and
   optimizer update are fused into a single compiled graph
2. **Functional gradients** — `nn.value_and_grad` returns a gradient dict,
   not in-place `.grad` attributes, which is what `mx.compile` expects
3. **Single eval boundary** — `mx.eval` is called *outside* the compiled
   function, after it returns, to force evaluation of the entire graph
4. **No Python conditionals inside** — the `if max_grad_norm > 0` check
   is on a Python float (config value), not a tensor, so it's safe
