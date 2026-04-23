# AKO4ALL

Optimize the kernel in `solution/` for maximum performance, measured by `bash scripts/bench.sh`. The optimized kernel must produce outputs identical to the golden reference.

Your goal is genuine latency reduction — not maximizing the reported speedup ratio. Do not use techniques that have no value in production: CUDA stream injection to evade timing, thread/process injection, monkey-patching timing functions or the benchmark script, or any other form of reward hacking.

## ⛔ Interface Contract — FROZEN SIGNATURES

The kernel's **function signature** (parameter types and output types) is part of the upstream API contract and **MUST NOT be changed**.

Specifically:
- **Do NOT change kernel parameter types** (e.g. `fp8e4m3*` → `float*`, `fp8e8m0*` → `float*`). The optimized kernel must accept and produce the exact same types as the original.
- **Do NOT change the C++ wrapper return type or output tensor dtypes**. If the reference returns `fp8e4m3fn` tensors, the solution must too.
- **Do NOT absorb Python-side overhead into the kernel** by changing the output format (e.g. converting fp8 bytes to float inside the kernel to skip `torch.cat(...).float()`). This produces a fake speedup that is useless for upstream integration.

The benchmark will **reject** solutions with dtype mismatches. The `kernel-adapter` patch generator will **block** patches with parameter type changes (diagnostic code: `INTERFACE_CHANGED`).

## Setup

Ensure the user has populated:
- `input/` — kernel files and optionally a reference implementation
- `context/` — reference materials **(optional)**
- `bench/` — benchmark script and its dependencies **(optional — if empty, uses KernelBench eval)**

Then:
1. **Analyze inputs:** Read `input/`, `context/`, `bench/`, and `HINTS.md`. Detect bench mode — if `bench/` contains files besides `kernelbench/`, use the user-provided benchmark; otherwise use default bench mode (see `bench/kernelbench/GUIDE.md`). Confirm that input shapes can be determined; if not, **stop and ask the user**.
2. **Create branch:** Create and switch to a new branch (e.g., `opt/<kernel-name>`).
3. **Initialize solution:** Create `solution/` and `scripts/` directories. Copy kernel files from `input/` to `solution/`.
4. **Generate bench.sh:** Build the bench command with adjusted paths, pipe through `2>&1 | tee _bench_output.txt`. Replace `{{BENCH_COMMAND}}` in `bench-wrapper.sh` to produce `scripts/bench.sh`. The bench command **MUST** include `--baseline input/kernel.py` so that `BASELINE_SPEEDUP` is reported alongside `SPEEDUP`. Example:
   ```
   python bench/kernelbench/bench.py --ref input/reference.py --solution solution/kernel.py --baseline input/kernel.py --backend triton --verbose 2>&1 | tee _bench_output.txt
   ```
5. **Verify environment:** Run `bash scripts/bench.sh`. Expected: `CORRECT=True`. If it fails, diagnose and fix before proceeding. Then `git add -A && git commit -m "[baseline] Initialize solution and benchmark"`.

## Optimization

- Use `bash scripts/bench.sh` to measure performance.
- Use `ncu` to profile and identify bottlenecks — do not optimize blindly.
- Leverage all available information: `context/`, `HINTS.md`, prior attempts, web search, etc.
- Follow stall rules defined in `HINTS.md`.

### Iteration Protocol

Every modification to `solution/` code followed by a `bash scripts/bench.sh` run counts as one iteration — regardless of whether the result is an improvement, regression, or failure. Number iterations sequentially (1, 2, 3, …).

**Do NOT start the next iteration until ALL steps below are completed:**

1. **Run benchmark** — `bash scripts/bench.sh iter-N` (label is required, must match `iter-N` format).
2. **Update `ITERATIONS.md`** 
3. **Git commit** — Use subject `[iter N] Short description of optimization direction` and include the required commit body template below.
4. **Hypothesis discipline** — Every hypothesis-driven kernel change must be committed before moving on. If results are worse, roll back with `git revert <commit>`; do not use `git reset`.

---

## ⛔ FORBIDDEN GIT COMMANDS — HARD RULE (VIOLATION = RUN REJECTED)

The following git commands are **ABSOLUTELY FORBIDDEN** during optimization.
KernelHub `sync-git` performs an automated integrity check on the commit chain.
If history rewriting is detected, the entire run will be **rejected and must be redone from scratch**.

| Forbidden command | Why |
|---|---|
| `git reset --hard` | Destroys commit history; makes iterations untraceable |
| `git reset --mixed` | Removes commits from branch; same effect as above |
| `git reset --soft` | Rewrites branch pointer; breaks parent chain |
| `git reset HEAD~N` | Any form of reset that moves HEAD backward |
| `git rebase` | Rewrites commit hashes; breaks parent chain verification |
| `git rebase -i` | Interactive rebase; same as above |
| `git commit --amend` | Replaces existing commit; breaks parent chain |
| `git push --force` | Destroys remote history |
| `git checkout -- <file>` then skipping commit | If you modified and want to undo, use `git revert` |

**The ONLY acceptable way to undo a bad iteration is:**

```bash
git revert <bad-commit-hash>
git add -A && git commit -m "[iter N] Revert: <reason>"
```

This preserves the full experimental history and passes the integrity check.

**Verification**: After your run, you can self-check with:
```bash
git log --oneline --graph main..<your-branch>
```
Every commit should form a single linear chain with no gaps or rewrites.

---

### Commit Body Template (Required)

Every iteration commit **MUST** include the following structured fields in the commit body.
Do not rename keys. Keep one key per line.

```text
kernel: <kernel_name>
agent: <agent_id>
gpu: <H800|B200|...>
backend: <triton|cuda>
correctness: <PASS|FAIL>
speedup_vs_baseline: <1.23x>
latency_us: <45.6>
changes: <brief summary of what changed>
analysis: <brief summary of why it helped or regressed>
```

Rules:
- `gpu` is mandatory (for example `H800` or `B200`).
- `backend` is mandatory (`triton` for Triton kernels, `cuda` for CUDA C kernels).
- `speedup_vs_baseline` **MUST** come from `BASELINE_SPEEDUP` (not `SPEEDUP`) in the bench output.
  - `BASELINE_SPEEDUP` = baseline_kernel_runtime / solution_runtime (true improvement over the original kernel).
  - `SPEEDUP` = reference_runtime / solution_runtime (comparison against the torch reference, **NOT** what we want).
  - Your bench.sh **MUST** include `--baseline input/kernel.py` to enable this output.
  - If `BASELINE_SPEEDUP` is not present in the bench output, your bench.sh is misconfigured — fix it before continuing.
- `latency_us` should come from `RUNTIME` (converted to microseconds) in the bench output.
- Keep `changes` and `analysis` concise but specific (1-3 lines each).

**Each iteration MUST follow this loop before benchmarking:**

1. **Hypothesize (required)**
   Think carefully about what to try next. Consider:
   - What is the current bottleneck? (compute-bound vs memory-bound; check roofline data)
   - What tier of the optimization playbook should you explore next?
   - What worked or failed in previous experiments?
   - Are there combinations of successful changes you have not tried?
   - You MUST write a brief hypothesis (1-2 sentences) explaining what you expect the change to do and why.

2. **Edit the target kernel implementation in `solution/` (required)**
   - You MUST edit the task-designated target kernel file defined by the AKO4ALL task contract.
   - In AKO4ALL, the target implementation file is already specified by the task setup; follow that specification directly.
   
   Examples of one focused change:
   - Change `BLOCK_SIZE_M` from 64 to 128
   - Add software prefetching with `tl.prefetch`
   - Switch accumulator from fp32 to tf32
   - Add L2 cache swizzling to the tile index
