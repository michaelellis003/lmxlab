# Unified Memory on Apple Silicon

Apple Silicon's unified memory architecture affects how ML frameworks
manage data. This page describes what unified memory means for lmxlab,
how it differs from discrete GPU setups, and what trade-offs it
introduces.

## What unified memory means

On traditional systems (NVIDIA GPUs), CPU and GPU have separate memory pools.
Moving data between them requires explicit copies across the PCIe bus:

```
┌─────────┐    PCIe     ┌─────────┐
│   CPU   │◄───────────►│   GPU   │
│  Memory │  (slow)     │  Memory │
│  (DDR5) │             │ (HBM/   │
│         │             │  GDDR)  │
└─────────┘             └─────────┘
```

On Apple Silicon, CPU and GPU share the same physical memory:

```
┌─────────────────────────────────┐
│        Unified Memory           │
│    (LPDDR5, shared by all)      │
│                                 │
│  ┌─────┐  ┌─────┐  ┌────────┐  │
│  │ CPU │  │ GPU │  │ Neural │  │
│  │cores│  │cores│  │ Engine │  │
│  └─────┘  └─────┘  └────────┘  │
└─────────────────────────────────┘
```

There is no copy. When the CPU writes an array, the GPU can read it
immediately (and vice versa). This removes the class of bugs and
performance costs associated with host-device transfers.

## What this means for lmxlab

### No device management

In PyTorch, forgetting `.to(device)` is a common source of errors:

```python
# PyTorch: must explicitly manage device placement
model = model.to('mps')       # Move model to GPU
x = x.to('mps')               # Move data to GPU
# RuntimeError if either is omitted
```

In lmxlab (MLX), there is no device concept:

```python
# MLX: everything lives in unified memory
model = LanguageModel(config)  # Already accessible to GPU
x = mx.array([[1, 2, 3]])     # Already accessible to GPU
logits, _ = model(x)          # Just works
```

The lmxlab codebase contains no `.to()`, `.cuda()`, `.cpu()`, or
`device=` calls. The hardware provides a single address space, so
device placement is unnecessary.

### Zero-copy data loading

On CUDA systems, data loading pipelines must carefully manage host-to-device
transfers, often using pinned memory and async copies. On unified memory,
the data is already where the GPU can access it:

```python
# No special data loading machinery needed
tokens = mx.array(tokenizer.encode(text), dtype=mx.int32)
# tokens is immediately usable by the GPU
```

This is why lmxlab's `batch_iterator` is a simple Python generator,
with no `DataLoader` using `pin_memory=True` and no `num_workers` for
parallel prefetching across a PCIe boundary.

### KV cache stays in place

During generation, the KV cache grows with each token. On discrete GPUs,
GPU memory must be managed carefully to avoid OOM. On unified
memory, the KV cache is just more arrays in the same memory pool:

```python
# No memory management needed; cache just grows
for _ in range(max_tokens):
    logits, cache = model(next_token, cache=cache)
    mx.eval(logits, *[c for pair in cache for c in pair])
```

## Trade-offs and constraints

Unified memory is not strictly better than discrete GPU memory. Here are
the important trade-offs:

### Memory bandwidth

| System | Memory Bandwidth |
|--------|-----------------|
| M1 Max | ~400 GB/s |
| M2 Ultra (192GB) | ~800 GB/s |
| M4 Max (128GB) | ~546 GB/s |
| NVIDIA A100 (80GB HBM) | ~2,039 GB/s |
| NVIDIA H100 (80GB HBM3) | ~3,350 GB/s |

Apple Silicon has much lower memory bandwidth than datacenter GPUs.
For large models where the bottleneck is weight loading (inference),
this means lower tokens/second. But for small-to-medium models that
fit in cache, the gap narrows significantly.

### Total memory capacity

| Chip | Max Unified Memory |
|------|-------------------|
| M1 | 16 GB |
| M1 Max | 64 GB |
| M2 Ultra | 192 GB |
| M3 Max | 128 GB |
| M4 Max | 128 GB |

The advantage: **all** of this memory is available to the model.
A 64GB M1 Max can load a ~30B parameter model in 4-bit quantization.
A 192GB M2 Ultra can run 70B+ models.

There is no separate "GPU memory" limit; the entire unified pool
is usable.

### No CPU offloading

On CUDA systems, PyTorch's `device_map='auto'` can split a model
across CPU and GPU memory, loading layers on demand. This is possible
because CPU memory is separate and typically much larger.

On unified memory, there is no separate CPU memory pool to offload to.
If the model exceeds unified memory, the only options are quantization
or a smaller model. There is no gradual degradation, only an OOM error.

**Mitigation strategies:**

1. Quantization: `quantize_model(model, bits=4)` reduces memory ~8x.
2. Smaller models: use config factories with smaller dimensions.
3. LoRA: fine-tune with frozen base weights (no optimizer state for
   frozen params).

### Lazy evaluation interaction

MLX's lazy evaluation interacts with unified memory in an important way:
computation graphs can grow large before evaluation, consuming memory
for intermediate results. The `mx.eval()` call forces evaluation and
frees intermediate buffers.

This is why lmxlab places `mx.eval()` at explicit boundaries:

```python
# Training: eval after each step
loss = self._step_fn(x, y)
mx.eval(loss, self.model.parameters(), self.optimizer.state)

# Generation: eval after each token
logits, cache = model(next_token, cache=cache)
mx.eval(logits, *[c for pair in cache for c in pair])
```

Without these boundaries, the graph accumulates and memory grows
unboundedly. See [MLX Idioms](mlx-idioms.md) for more details.

## Capacity estimates

Approximate model sizes for different Apple Silicon chips:

| Chip (Memory) | FP16 Model | 4-bit Model |
|---------------|-----------|-------------|
| M1 (16GB) | ~7B | ~28B |
| M1 Max (64GB) | ~30B | ~120B |
| M2 Ultra (192GB) | ~90B | ~360B |
| M4 Max (128GB) | ~60B | ~240B |

These are rough upper bounds. Actual usable memory is less because
the OS, other apps, and inference overhead (KV cache, activations)
consume memory too. A practical rule of thumb: assume 70-80% of
unified memory is available for model weights.

## Summary

| Aspect | Discrete GPU (CUDA) | Unified Memory (MLX) |
|--------|-------------------|---------------------|
| Device management | `.to('cuda')` everywhere | None (no device concept) |
| Data transfer | PCIe copies, pinned memory | Zero-copy |
| Memory capacity | GPU VRAM (24-80GB typical) | Full unified pool (16-192GB) |
| Bandwidth | Very high (2-3 TB/s HBM) | Lower (400-800 GB/s LPDDR) |
| CPU offloading | `device_map='auto'` | Not possible |
| Programming model | Manage two memory spaces | One memory space |
| OOM recovery | Offload to CPU | Quantize or use smaller model |

The unified memory model trades raw bandwidth for simplicity. For
educational and research use (lmxlab's target), the removal of
device management complexity reduces both code size and bug surface.
