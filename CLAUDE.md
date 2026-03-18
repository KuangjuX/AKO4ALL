# kernel-opt-agent — Developer Guide

This is the **template repository** for spawning isolated GPU kernel optimization environments.
It is NOT an optimization environment itself — use `setup.sh` to create one.

## Repository Purpose

Provides scaffolding to create child environments where a Code Agent (e.g. Claude Code)
autonomously optimizes a GPU kernel (Triton or CUDA), using the flashinfer-bench SDK for evaluation.

Supports any flashinfer-bench compatible dataset, including user-defined operators.

## Directory Structure

```
kernel-opt-agent/
├── CLAUDE.md                          # This file (developer guide for the template repo)
├── README.md                          # Human-facing usage guide
├── setup.sh                           # Creates isolated child environments
├── .gitignore                         # Git ignore rules (copied to children)
├── templates/
│   ├── task.md                        # Main task template (assembled into CLAUDE.md at spawn)
│   ├── hints.md                       # Default HINTS.md template (copied to children)
│   ├── fragments/
│   │   ├── objective-scratch.md       # Objective for scratch mode
│   │   ├── objective-existing.md      # Objective for existing mode
│   │   ├── gpu-a100.md                # A100 hardware specs
│   │   ├── gpu-b200.md                # B200 hardware specs
│   │   ├── backend-local.md           # Local execution instructions
│   │   └── backend-modal.md           # Modal execution instructions
│   └── agent/
│       └── claude.json                # Claude Code agent config (output filenames + permissions)
└── scripts/
    ├── generate_context.py            # Extracts operator metadata for template rendering
    ├── bench_utils.py                 # Shared benchmark utilities (baseline caching, scoring)
    ├── bench.sh                       # Local benchmark script
    ├── bench_modal.sh                 # Modal benchmark script
    ├── run_local.py                   # Local A100 benchmark runner
    ├── run_modal.py                   # Modal B200 benchmark runner
    └── pack_solution.py               # Pack kernel for evaluation
```

## How setup.sh Works

1. Parses `--operator`, `--mode`, `--backend`, `--language`, `--gpu`, `--agent`, `--kernel`, `--name`, `--dataset`, `--task`, `--hints`
2. If `--operator` not provided, lists available operators from the dataset and exits
3. Derives `LANGUAGE_NAME` from `--language` (triton→Triton, cuda→CUDA) and resolves GPU (auto-detect for local, required for modal)
4. Reads agent config from `templates/agent/{agent}.json` for output filenames and permissions
5. Auto-discovers operator definition from `<dataset>/definitions/*/<operator>.json`
6. Runs `generate_context.py` to extract template variables (shapes, workload summary, etc.)
7. Auto-increments run number -> creates `kernel-opt-agent-run-NNN[-label]/`
8. Copies definition.json and workloads.jsonl from the dataset
9. Auto-generates `config.toml` with operator name, language, GPU, and default build settings
10. Copies backend-specific bench script and runner (`run_local.py` or `run_modal.py`)
11. Generates `.claude/settings.local.json` from agent config permissions
12. Extracts reference kernel from definition.json (scratch) or copies user-provided kernel (existing)
13. Copies hints file (from `--hints` or `templates/hints.md`) as `HINTS.md`
14. Assembles task file from `task.md` + fragment placeholders -> output filename from agent config
15. Initializes a git repo in the child directory

## Template Placeholders

Templates use these placeholders (replaced at spawn time):

| Placeholder | Source | Example |
|-------------|--------|---------|
| `{{LANGUAGE}}` | `--language` flag (default: triton) | `triton` |
| `{{LANGUAGE_NAME}}` | derived from `--language` | `Triton` |
| `{{GPU}}` | `fragments/gpu-{gpu}.md` content (or fallback `- GPU: {NAME}`) | (hardware specs line) |
| `{{GPU_NAME}}` | `uppercase(--gpu)` | `A100`, `H100` |
| `{{BACKEND}}` | `fragments/backend-{backend}.md` content | (execution instructions) |
| `{{OBJECTIVE}}` | `fragments/objective-{mode}.md` content | (multi-line) |
| `{{OPERATOR}}` | definition name | `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64` |
| `{{OPERATOR_DESCRIPTION}}` | first sentence of definition description | `Batched Native Sparse Attention (DSA)...` |
| `{{NUM_WORKLOADS}}` | count of workloads | `23` |
| `{{SHAPE_SUMMARY}}` | auto-formatted input/output shapes | `Key shapes: ...` |
| `{{WORKLOAD_SUMMARY}}` | distribution of variable axes | `Workloads: num_tokens in {1(x1),...}` |

## How to Modify Templates

- **Task template**: Edit `templates/task.md` — the main template assembled into the child's task file
- **Objective fragments**: Edit `templates/fragments/objective-scratch.md` or `objective-existing.md`
- **GPU hardware specs**: Edit or add `templates/fragments/gpu-{name}.md` (optional; if missing, `{{GPU}}` falls back to `- GPU: {NAME}`)
- **Backend execution instructions**: Edit `templates/fragments/backend-local.md` or `backend-modal.md`
- **Agent config**: Edit `templates/agent/claude.json` (output filenames + permissions)
- **Hints template**: Edit `templates/hints.md`
- **Config defaults**: Edit the `config.toml` generation block in `setup.sh`
- **Benchmark scripts**: Edit `scripts/bench.sh`, `scripts/bench_modal.sh`, etc.

## Baseline Caching

The benchmark system caches reference implementation performance on the first run:
- First `bench.sh` run profiles the reference (20 iterations) and saves metrics to `baseline.json`
- Subsequent runs skip reference profiling and use cached values with 100 iterations for the solution
- This provides stable reference metrics and more accurate solution timing
- Use `--force-baseline` to re-profile: `bash scripts/bench.sh --force-baseline`
- `baseline.json` auto-invalidates when workloads change

## Key Constraints

- Agents only edit `solution/{language}/kernel.py` (and optionally `config.toml`)
- `config.toml`: `destination_passing_style = false` by default
- Operator data (definition.json, workloads.jsonl, reference kernel) comes from the dataset at spawn time, not from static files in this repo
