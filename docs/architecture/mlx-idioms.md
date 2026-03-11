# MLX Idioms

MLX looks like PyTorch at first glance -- `nn.Module`, `nn.Linear`,
`mx.array` -- but the execution model is fundamentally different. This page
covers the MLX patterns that lmt-metal relies on and how they differ from
their PyTorch equivalents.

## Lazy evaluation and `mx.eval`

This is the single most important difference. In PyTorch, every operation
executes immediately. In MLX, operations build a computation graph that
is evaluated lazily.

```python
import mlx.core as mx

a = mx.ones((1000, 1000))
b = mx.ones((1000, 1000))
c = a + b  # No computation happens here -- c is a graph node

mx.eval(c)  # NOW the addition runs on the GPU
```

This matters because MLX can fuse operations and optimize the graph before
executing it. But it also means you need to be deliberate about when
evaluation happens.

**The rule in lmt-metal:** call `mx.eval` at explicit boundaries -- after
a training step, after generation produces a token, after evaluation
computes a loss. Do not scatter `mx.eval` calls inside model code.

```python
# In the Trainer:
loss = self._step_fn(x, y)
mx.eval(loss, self.model.parameters(), self.optimizer.state)
# ^--- One eval boundary per training step, covering all outputs
```

If you forget `mx.eval`, the graph keeps growing and memory usage climbs.
If you call it too often, you break fusion opportunities and hurt
performance. The training loop in `Trainer` demonstrates the right balance.

## `nn.value_and_grad` (not `.backward()`)

PyTorch uses imperative autograd: call `loss.backward()`, then read
`.grad` attributes on parameters. MLX uses a functional approach borrowed
from JAX.

=== "MLX (lmt-metal)"

    ```python
    import mlx.nn as nn

    # Create a function that computes loss given the model
    def loss_fn(model, x, y):
        logits, _ = model(x)
        return nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1),
            reduction='mean',
        )

    # nn.value_and_grad returns a function that computes
    # both the loss AND gradients w.r.t. model parameters
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # Call it: returns (loss_value, gradient_dict)
    loss, grads = loss_and_grad_fn(model, x, y)
    ```

=== "PyTorch (for comparison)"

    ```python
    loss = loss_fn(model, x, y)
    loss.backward()           # Mutates parameter .grad attributes
    grads = {name: p.grad for name, p in model.named_parameters()}
    ```

The MLX approach is functional: `loss_and_grad_fn` is a pure function that
returns gradients as a dictionary without mutating anything. This makes it
straightforward to compose with `mx.compile`.

## `mx.compile` for training steps

`mx.compile` traces a function and produces an optimized version. In
lmt-metal, the entire training step (forward + backward + optimizer update)
is compiled:

```python
if config.compile_step:
    self._step_fn = mx.compile(
        self._single_step,
        inputs=model.trainable_parameters(),
        outputs=model.trainable_parameters(),
    )
```

The `inputs` and `outputs` arguments tell the compiler which state is
read and written by the function. This is necessary because the model
parameters are mutated in-place by the optimizer.

!!! warning "Compile gotcha"
    `mx.compile` traces the function once and caches the graph. If your
    function has Python-level control flow that depends on tensor values
    (e.g., `if loss > threshold:`), the condition is captured at trace time
    and will not change on subsequent calls. Keep compiled functions free
    of data-dependent branches.

**When not to compile:** During debugging, set `compile_step=False` in
`TrainConfig` to get clearer error messages and stack traces.

## Unified memory: no `.to(device)`

On Apple Silicon, CPU and GPU share the same physical memory. MLX exploits
this -- there is no concept of device placement.

=== "MLX"

    ```python
    import mlx.core as mx

    x = mx.random.normal((32, 512))
    # x is already accessible to both CPU and GPU
    # No .to('mps') or .cuda() needed
    ```

=== "PyTorch"

    ```python
    import torch

    x = torch.randn(32, 512)
    x = x.to('mps')  # Copy to Apple GPU memory
    model = model.to('mps')  # Copy all parameters
    ```

This eliminates an entire category of bugs (tensors on different devices)
and simplifies the code. In lmt-metal, you will never see a `.to()` call.

The tradeoff: you cannot have separate CPU and GPU memory pools. If your
model and data together exceed unified memory, you are out of luck (there
is no automatic CPU offloading like PyTorch's `device_map='auto'`).

## `mx.fast.scaled_dot_product_attention`

MLX provides a fused attention kernel that is substantially faster than
manual Q @ K^T / sqrt(d) @ V. lmt-metal uses it in all attention modules:

```python
out = mx.fast.scaled_dot_product_attention(
    q, k, v, scale=self.scale, mask=mask
)
```

This is analogous to PyTorch's `F.scaled_dot_product_attention`, but
the MLX version is designed for the Apple GPU's tile-based architecture.
It handles the softmax, scaling, and masking in a single fused operation.

The expected tensor shapes are:

- `q`: `(batch, n_heads, seq_len, head_dim)`
- `k`: `(batch, n_kv_heads, kv_len, head_dim)`
- `v`: `(batch, n_kv_heads, kv_len, head_dim)`
- `mask`: broadcastable to `(batch, n_heads, seq_len, kv_len)` or `None`

When `n_kv_heads < n_heads` (GQA), `mx.fast.scaled_dot_product_attention`
handles the head broadcasting internally.

## Optimizer updates: functional, not in-place

In PyTorch, you call `optimizer.step()` and it mutates parameters in-place.
In MLX, the optimizer produces new parameter values:

```python
# MLX pattern (used in Trainer._single_step):
loss, grads = self._loss_and_grad(self.model, x, y)

if self.config.max_grad_norm > 0:
    grads, _ = optim.clip_grad_norm(grads, max_norm=self.config.max_grad_norm)

self.optimizer.update(self.model, grads)
# model parameters are updated through the optimizer
```

Gradient clipping is also functional: `optim.clip_grad_norm` returns a new
gradient dict rather than modifying the input.

## `nn.RoPE` and other built-in modules

MLX provides several commonly-used components out of the box. lmt-metal wraps
them for registry compatibility but delegates to the MLX implementations:

- `nn.RoPE` -- Rotary Position Embedding
- `nn.RMSNorm` -- Root Mean Square Normalization
- `nn.LayerNorm` -- Layer Normalization
- `nn.SinusoidalPositionalEncoding` -- Sinusoidal position encoding
- `nn.ALiBi` -- Attention with Linear Biases

Using MLX's built-in modules means we get upstream performance improvements
for free, and the code stays close to the mathematical definitions.

## Summary: MLX vs PyTorch mental model

| Concept | PyTorch | MLX |
|---------|---------|-----|
| Evaluation | Eager (immediate) | Lazy (call `mx.eval`) |
| Gradients | `loss.backward()` + `.grad` | `nn.value_and_grad` returns dict |
| Compilation | `torch.compile` (optional) | `mx.compile` (opt-in, explicit I/O) |
| Device | `.to('cuda')` / `.to('mps')` | Unified memory, no device concept |
| Fused attention | `F.scaled_dot_product_attention` | `mx.fast.scaled_dot_product_attention` |
| Optimizer | `optimizer.step()` (in-place) | `optimizer.update(model, grads)` |

The common thread: MLX favors **explicit, functional** patterns over
**implicit, mutation-based** ones. This requires a slight adjustment in
thinking but produces code that is easier to reason about and compose.
