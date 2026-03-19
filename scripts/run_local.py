"""
FlashInfer-Bench Local Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks locally.
Caches reference baseline on first run for stable, efficient subsequent runs.
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet
from scripts.bench_utils import extract_results, run_and_report
from scripts.pack_solution import pack_solution


def get_trace_set_path() -> str:
    """Get trace set path from config.toml or environment variable."""
    # Prefer config.toml (written at spawn time)
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    config_path = PROJECT_ROOT / "config.toml"
    if config_path.exists():
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
        path = config.get("build", {}).get("dataset_path")
        if path:
            return path
    # Fall back to environment variable
    path = os.environ.get("FIB_DATASET_PATH")
    if not path:
        raise EnvironmentError(
            "Dataset path not found. Either set 'dataset_path' in config.toml "
            "or the FIB_DATASET_PATH environment variable."
        )
    return path


def run_benchmark(solution: Solution, config: BenchmarkConfig) -> dict:
    """Run benchmark locally and return results."""
    trace_set_path = get_trace_set_path()
    trace_set = TraceSet.from_path(trace_set_path)

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])

    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    benchmark = Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=True)
    return extract_results(result_trace_set, definition.name)


def main():
    """Pack solution and run benchmark."""
    parser = argparse.ArgumentParser(description="Run benchmark with optional trajectory tracking")
    parser.add_argument("--label", type=str, default=None, help="Optional label for trajectory tracking")
    parser.add_argument("--force-baseline", action="store_true",
                        help="Force re-profiling of reference baseline")
    args = parser.parse_args()

    print("Packing solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")

    run_and_report(
        solution, run_benchmark,
        force_baseline=args.force_baseline,
        label=args.label,
        backend="local",
    )


if __name__ == "__main__":
    main()
