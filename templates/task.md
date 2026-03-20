# Kernel Optimization Task

You are a GPU kernel optimization expert. Your task is to optimize the kernel in `solution/` for maximum performance.

Read `HINTS.md` before starting for user-provided hints and constraints.

## Kernel

- **Language**: Triton
- **Entry point**: `solution/kernel.py` -> `softmax_kernel()` function
- **Functionality**: Fused softmax over the last dimension of a 2D tensor
- **Inputs**: `x` (float32 tensor, shape `[M, N]`)
- **Outputs**: `out` (float32 tensor, shape `[M, N]`)
- **Computation pattern**: Row-wise reduction (max + sum for numerical stability), then elementwise exp and normalize
- **Output allocation**: The `run()` function allocates and returns output tensors

## Benchmark

```bash
bash bench.sh [label]
```

- Per-case results with latency and correctness status
- **PASSED** = correct output within tolerance. **FAILED** = incorrect or crash
- Primary metric: mean speedup across all workloads (higher is better)
- Label runs for tracking: `bash bench.sh "v1_tiling"` saves to `trajectory/`

### Correctness Tolerance

- atol=1e-3, rtol=1e-3: `|actual - expected| < atol + rtol * |expected|`
- Moderately strict — bf16 accumulation may fail on large N; prefer fp32 accumulation

### Benchmark Internals

- Timing: CUDA events with synchronization, L2 cache warm (no flushing)
- Config: 5 warmup, 100 iterations, 3 trials, median per trial, mean across trials

### Workload Distribution

6 workloads: M=128/N=512, M=256/N=1024, M=1024/N=4096, M=2048/N=4096, M=4096/N=4096, M=4096/N=8192

## File Rules

- `solution/` — optimization target, benchmarked. Editable.
- `bench/` and `bench.sh` — benchmark infrastructure. Read-only.
- `scripts/` — workspace for your profiling/debug tools. Editable.
