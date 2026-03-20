# AutoKernelForge — Session 1 Action Guide

You are operating in the **template repository**. Your job is to gather information from the user, analyze their kernel and benchmark script, and create an isolated child environment for optimization.

## Conversation Protocol

### 1. Gather inputs

Ask the user for:
- **(a) Kernel path** — file or directory containing the kernel to optimize
- **(b) Bench script path** — the script that benchmarks the kernel
- **(c) Hints** (optional) — a file with optimization hints, or verbal hints to include
- **(d) Reference implementation** (optional) — a reference implementation that benchmark uses for correctness checking

If the user provides all upfront, skip asking.

### 2. Analyze kernel and bench script

Read the kernel and bench script (including any libraries/configs they import). Your goal is to understand them well enough to:
- Write an accurate kernel description for Session 2
- Construct the correct bench command with adjusted paths
- Identify any critical information Session 2 needs (correctness tolerance, timing method, workload parameters, edge cases, etc.)

Use your judgment on what's important — don't mechanically check every possible dimension. Focus on what would actually affect optimization decisions.

### 3. Confirm with user

Present findings. Ask the user to confirm or correct:
- Kernel summary (language, entry point, functionality)
- Bench invocation and how paths will be adjusted in the child env
- Anything that couldn't be inferred (optimization goal, GPU, environment name, etc.)
- Optional: a short label for the run directory

### 4. Create child environment

Follow the procedure in [Child Environment Creation](#child-environment-creation) below.

### 5. Done

Tell the user:
```
cd <child-dir> && claude
```

## Child Environment Creation

### Naming

Create the child directory as a **sibling** of this repo (i.e., `../<name>`):
- With label: `akf-run-{label}`
- Without: `akf-run-{YYYYMMDD_HHMMSS}`

### Directory structure

```
akf-run-xxx/
├── CLAUDE.md                   # Generated task spec (self-contained)
├── HINTS.md                    # Optimization hints
├── bench.sh                    # Benchmark wrapper (read-only)
├── .gitignore
├── solution/                   # Kernel files (editable, benchmarked)
├── bench/                      # Bench script + deps (read-only)
├── scripts/                    # Agent workspace for profiling/debug tools
└── .claude/
    └── settings.local.json     # Agent permissions
```

### Steps

1. **Create directory tree**: `mkdir -p <child>/{solution,bench,scripts,.claude}`
   - `scripts/` is an empty workspace for the agent to create profiling/debug tools during optimization

2. **Copy kernel files to `solution/`**: Copy the kernel file(s) into `solution/`. Preserve filenames. If it's a directory, copy contents.

3. **Copy bench script + local deps to `bench/`**: Copy the bench script and any local files it imports into `bench/`. If the bench script is in a directory with helpers, copy the directory contents. Do NOT modify the bench script.

4. **Copy reference implementation to `bench/`** (if provided): Copy the reference implementation into `bench/`. Adjust the bench command to point to it.

5. **Adjust bench command for new paths**: Figure out the command to run the bench script from the child root, with paths adjusted:
   - Kernel path becomes `solution/<filename>`
   - Bench script path becomes `bench/<filename>`
   - Example: if original was `python bench.py --kernel my_kernel.py`, becomes `python bench/bench.py --kernel solution/my_kernel.py`
   - Pipe through tee to capture output: `<command> 2>&1 | tee _bench_output.txt`

6. **Generate `bench.sh`**: Read `templates/bench-wrapper.sh` from this repo, replace `{{BENCH_COMMAND}}` with the adjusted command from step 5, write to `<child>/bench.sh`, make executable.

7. **Write `.claude/settings.local.json`**: Copy `templates/agent/claude.json` from this repo to `<child>/.claude/settings.local.json`.

8. **Copy `.gitignore`**: Copy this repo's `.gitignore` to `<child>/.gitignore`.

9. **Write `HINTS.md`**: If user provided a hints file, copy it. If user gave verbal hints, write them. Otherwise copy `templates/hints.md` from this repo.

10. **Write `CLAUDE.md`**: Generate the child's CLAUDE.md. See [guidance below](#child-claudemd-guidance). Use `templates/task.md` as a reference example for format.

11. **Initialize git**: `cd <child> && git init && git add -A && git commit -m "Initial environment"`

### Child CLAUDE.md Guidance

The generated CLAUDE.md must be **self-contained** — Session 2 will not have access to this template repo.

**Always include:**
- **Role** — one line: "You are a GPU kernel optimization expert. Your task is to optimize the kernel in `solution/` for maximum performance."
- **Hints** — "Read `HINTS.md` before starting for user-provided hints and constraints."
- **Kernel** — language, entry point, what it computes, input/output shapes and types. Be specific enough that Session 2 can understand the code without guessing.
- **Benchmark** — how to run (`bash bench.sh [label]`), output format, what PASSED/FAILED means, primary metric.
- **File rules** — `solution/` is the optimization target (benchmarked). `bench/` and `bench.sh` are read-only. `scripts/` is a workspace for profiling/debug tools.

**Include when relevant** — any critical information you discovered during analysis that would affect optimization decisions. Examples: correctness tolerance (and what it implies for precision tradeoffs), timing method, workload distribution, known limitations of the benchmark. Don't pad with information that won't change what Session 2 does.

**Do NOT include** a fixed workflow or step-by-step optimization instructions. Session 2 is a capable optimization agent — let it decide its own approach.

## Constraints

- Never modify files outside the child directory.
- The child CLAUDE.md must be self-contained (Session 2 does not read this template repo).
- Always init git in the child.
- Adjust all paths in the bench command so they work from the child root directory.
- If the bench script has complex dependencies (conda env, pip packages, etc.), document the environment setup in the child CLAUDE.md's Benchmark section.
