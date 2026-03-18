# kernel-opt-agent

A tool for automated GPU kernel optimization using Code Agents (e.g. Claude Code). Built on the [flashinfer-bench](https://github.com/flashinfer-ai/flashinfer-bench) SDK, it supports any flashinfer-bench compatible dataset, including user-defined operators.

## Quick Start

```bash
# List available operators
bash setup.sh

# Create an optimization environment (local, auto-detect GPU, Triton, from scratch)
bash setup.sh --operator dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64

# Enter the child environment and start your code agent
cd ../kernel-opt-agent-run-001
claude
```

## Setup Options

```bash
bash setup.sh \
  --operator <name> \
  [--dataset /path/to/dataset] \    # default: $FIB_DATASET_PATH
  [--mode scratch|existing] \       # default: scratch
  [--backend local|modal] \         # default: local
  [--language triton|cuda] \        # default: triton
  [--gpu <name>] \                  # local: auto-detected; modal: required
  [--agent claude] \                # default: claude
  [--kernel /path/to/kernel.py] \   # required for existing mode
  [--name <label>] \                # optional label for the run
  [--task /path/to/task.md] \       # custom task template
  [--hints /path/to/hints.md]       # custom hints for the agent
```

### Examples

```bash
# Local, auto-detect GPU, from scratch, with label
bash setup.sh --operator gdn_decode_qk4_v8_d128_k_last --name "gdn_exp_1"

# Local with explicit GPU
bash setup.sh --operator gdn_decode_qk4_v8_d128_k_last --gpu h100

# Local, from existing kernel
bash setup.sh --operator dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64 \
    --mode existing --kernel /path/to/kernel.py

# Modal B200, from scratch (--gpu required for modal)
bash setup.sh --operator moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048 \
    --backend modal --gpu b200 --name "moe_b200"

# Custom dataset
bash setup.sh --operator my_custom_op --dataset /path/to/my/traceset

# Custom task template and hints
bash setup.sh --operator dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64 \
    --task ./my_task.md --hints ./my_hints.md

# Deprecated flags still work (--prompt → --task, --info → --hints)
bash setup.sh --operator my_op --prompt ./my_task.md --info ./my_hints.md
```

## How It Works

1. `setup.sh --operator <name>` discovers the operator definition and workloads from the dataset
2. Creates a child environment `kernel-opt-agent-run-NNN[-label]/` with everything the agent needs
3. You start your code agent (e.g. `claude`) in the child environment
4. The agent reads `CLAUDE.md` and `HINTS.md`, then iteratively optimizes the kernel

### Child Environment Structure

```
kernel-opt-agent-run-NNN/
├── CLAUDE.md                   # Auto-generated task spec (operator details + instructions)
├── HINTS.md                    # User-editable hints and constraints
├── config.toml                 # Operator configuration
├── .gitignore
├── docs/
│   ├── definition.json         # Operator definition
│   └── workloads.jsonl         # Benchmark workloads
├── solution/
│   └── triton/                 # or cuda/, based on --language
│       └── kernel.py           # The kernel to optimize
├── scripts/
│   ├── bench_utils.py          # Shared benchmark utilities
│   ├── bench.sh                # Benchmark script
│   ├── run_local.py/run_modal.py  # Benchmark runner
│   └── pack_solution.py        # Pack kernel for evaluation
├── baseline.json               # Auto-created, cached reference performance
├── trajectory/                 # Auto-created, stores each bench run
└── .claude/
    └── settings.local.json     # Permission fallback
```

## Customization

### Custom Task Template

The default task template (`templates/task.md`) defines the agent's instructions. Override it with `--task`:

```bash
bash setup.sh --operator my_op --task ./my_task.md
```

### Custom Hints

`HINTS.md` is placed in each child environment for ad-hoc hints. Override the template with `--hints`:

```bash
bash setup.sh --operator my_op --hints ./my_analysis.md
```

Or edit `HINTS.md` directly in the child environment before launching.

### Custom Dataset

Any flashinfer-bench compatible trace set works. The expected structure:

```
dataset/
├── definitions/
│   └── <type>/
│       └── <operator>.json
└── workloads/
    └── <type>/
        └── <operator>.jsonl
```

## Benchmark Output

The benchmark script prints per-workload results and a final score:

```
operator_name:
  Workload 0c23b10...: PASSED | 0.492 ms | 4.17x speedup | abs_err=1.2e-03, rel_err=2.1e-04
  ...

══════════════════════════════════════════════════
  FINAL SCORE (mean speedup): 4.17x
  Passed: 23/23 | Min: 2.10x | Max: 8.34x
  By num_tokens:  1→4.17x  2→3.85x  ...
══════════════════════════════════════════════════
```

The `score` field (= `FINAL SCORE` = mean speedup across all workloads) is the primary metric. All workloads must PASS for a valid score.

### Baseline Caching

The first benchmark run profiles the reference implementation and caches the result to `baseline.json`. Subsequent runs skip reference profiling for faster, more stable comparisons and use higher iteration counts (100 vs 20) for more accurate solution timing.

```bash
bash scripts/bench.sh "baseline"          # First run: profiles reference, caches baseline
bash scripts/bench.sh "opt_v1"            # Uses cached baseline, 100 iterations for solution
bash scripts/bench.sh --force-baseline    # Force re-profile reference
```

## Requirements

**Local backend (default):**
- NVIDIA GPU (auto-detected via `nvidia-smi`, or specify with `--gpu`)
- Conda env `fi-bench` (Python 3.12, PyTorch 2.9.1+cu128, `flashinfer-bench`)
- `FIB_DATASET_PATH` pointing to the trace set, or use `--dataset`

**Modal backend:**
- `--gpu` is required (e.g. `--gpu b200`, `--gpu h100`)
- Conda env `fi-bench` (with `modal` and `flashinfer-bench`)
- Modal authenticated (`conda run -n fi-bench --no-capture-output modal setup`)
- Modal volume `flashinfer-trace` with the trace set uploaded
