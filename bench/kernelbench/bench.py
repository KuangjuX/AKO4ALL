#!/usr/bin/env python3
"""
Self-contained KernelBench benchmark for AKO4ALL.

Evaluates an optimized kernel (solution) against a reference kernel.
Inlines core logic from KernelBench's eval.py and timing.py so no external
KernelBench dependency is needed.

Usage:
    python bench/kernelbench/bench.py --ref input/kernel.py --solution solution/kernel.py [options]

Output (structured, one per line):
    COMPILED: True/False
    CORRECT: True/False
    RUNTIME: <ms>
    REF_RUNTIME: <ms>
    SPEEDUP: <x>
    BASELINE_RUNTIME: <ms>       (only when --baseline is provided)
    BASELINE_SPEEDUP: <x>        (only when --baseline is provided)

Portions of this file are derived from KernelBench
(https://github.com/ScalingIntelligence/KernelBench).

Copyright (c) 2023 Anne Ouyang, Simon Guo, Azalia Mirhoseini

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import argparse
import importlib
import os
import re
import statistics
import sys
import tempfile
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from io import StringIO
from typing import Any, Optional, Union

import torch
import torch.nn as nn


###############################################################################
# Inlined from KernelBench — timing utilities
###############################################################################


def clear_l2_cache(device: torch.device | str = "cuda"):
    """Clear L2 cache by thrashing with a large tensor (~256 MB)."""
    dummy = torch.empty((32, 1024, 1024), dtype=torch.int64, device=device)
    dummy.fill_(42)
    del dummy


def time_execution_with_cuda_event(
    kernel_fn: callable,
    args: list[Any],
    num_warmup: int = 3,
    num_trials: int = 10,
    discard_first: int = 1,
    verbose: bool = True,
    device: torch.device = None,
) -> list[float]:
    """
    Time a CUDA kernel over multiple trials using torch.cuda.Event.
    Measures cold-cache performance (L2 thrashed before each trial).

    Returns list of elapsed times in milliseconds.
    """
    if device is None:
        device = torch.cuda.current_device()

    with torch.cuda.device(device):
        # Warm ups
        for _ in range(num_warmup):
            kernel_fn(*args)
            torch.cuda.synchronize(device=device)

        torch.cuda.empty_cache()

        print(
            f"[Profiling] Using device: {device} {torch.cuda.get_device_name(device)}, "
            f"warm up {num_warmup}, trials {num_trials}"
        )

        elapsed_times: list[float] = []

        for trial in range(num_trials + discard_first):
            torch.cuda.synchronize(device=device)

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            clear_l2_cache(device=device)

            start_event.record()
            _ = kernel_fn(*args)
            end_event.record()

            torch.cuda.synchronize(device=device)

            elapsed_time_ms = start_event.elapsed_time(end_event)

            if trial >= discard_first:
                if verbose:
                    logical_idx = trial - discard_first + 1
                    print(f"Trial {logical_idx}: {elapsed_time_ms:.3g} ms")
                elapsed_times.append(elapsed_time_ms)

    return elapsed_times


def time_execution_with_host_time(
    kernel_fn: callable,
    args: list[Any],
    num_warmup: int = 3,
    num_trials: int = 10,
    discard_first: int = 1,
    verbose: bool = True,
    device: torch.device | None = None,
) -> list[float]:
    """
    Time a CUDA kernel using host-side wall-clock time (perf_counter).
    Includes Python overhead, launch costs, and synchronization.

    Returns list of elapsed times in milliseconds.
    """
    if device is None:
        device = torch.cuda.current_device()

    for _ in range(num_warmup):
        kernel_fn(*args)
        torch.cuda.synchronize(device=device)

    print(
        f"[Profiling] Using device: {device} {torch.cuda.get_device_name(device)}, "
        f"warm up {num_warmup}, trials {num_trials}"
    )

    torch.cuda.empty_cache()
    elapsed_times = []

    for trial in range(num_trials + discard_first):
        torch.cuda.synchronize(device=device)
        clear_l2_cache(device=device)

        start_time = time.perf_counter()
        kernel_fn(*args)
        torch.cuda.synchronize(device=device)
        end_time = time.perf_counter()

        elapsed_time_ms = (end_time - start_time) * 1000
        if trial >= discard_first:
            if verbose:
                logical_idx = trial - discard_first + 1
                print(f"Trial {logical_idx}: {elapsed_time_ms:.3g} ms")
            elapsed_times.append(elapsed_time_ms)

    return elapsed_times


def get_timing_function(method: str = "cuda_event") -> callable:
    """
    Return timing function by method name.

    Available: "cuda_event" (default), "host_time".
    """
    print(f"[Profiling] Using timing method: {method}")
    match method:
        case "cuda_event":
            return time_execution_with_cuda_event
        case "host_time":
            return time_execution_with_host_time
        case _:
            raise ValueError(
                f"Unsupported timing method: {method}. "
                f"Available: cuda_event, host_time"
            )


def get_timing_stats(elapsed_times: list[float], device: torch.device = None) -> dict:
    """Compute mean/std/min/max from a list of elapsed times (ms)."""
    mean_val = statistics.mean(elapsed_times)
    std_val = statistics.stdev(elapsed_times) if len(elapsed_times) > 1 else 0.0

    stats = {
        "mean": float(f"{mean_val:.3g}"),
        "std": float(f"{std_val:.3g}"),
        "min": float(f"{min(elapsed_times):.3g}"),
        "max": float(f"{max(elapsed_times):.3g}"),
        "num_trials": len(elapsed_times),
    }

    if device:
        stats["hardware"] = torch.cuda.get_device_name(device=device)
        stats["device"] = str(device)

    return stats


###############################################################################
# Inlined from KernelBench — eval utilities
###############################################################################


@dataclass
class KernelExecResult:
    """Result of a single kernel evaluation."""

    compiled: bool = False
    correctness: bool = False
    metadata: dict = field(default_factory=dict)
    runtime: float = -1.0  # ms
    runtime_stats: dict = field(default_factory=dict)
    ref_runtime: float = -1.0  # ms
    ref_runtime_stats: dict = field(default_factory=dict)


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_tolerance_for_precision(precision: Union[str, torch.dtype]) -> float:
    """
    Tolerance for correctness checks, inspired by torchbench.

    fp32: 1e-4, fp16/bf16: 1e-2.
    """
    if isinstance(precision, str):
        dtype_map = {
            "fp32": torch.float32,
            "float32": torch.float32,
            "fp16": torch.float16,
            "float16": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
        }
        precision = dtype_map[precision]

    tolerances = {
        torch.float32: 1e-4,
        torch.float16: 1e-2,
        torch.bfloat16: 1e-2,
    }
    assert precision in tolerances, f"Unsupported precision: {precision}"
    return tolerances[precision]


def _process_input_tensor(
    inp, device, backend="cuda", precision=torch.float32
):
    """Move tensor to device with correct dtype. Non-tensors pass through."""
    if not isinstance(inp, torch.Tensor):
        return inp
    if inp.is_floating_point():
        return inp.to(dtype=precision, device=device)
    return inp.to(device=device)


def get_error_name(e: Exception) -> str:
    return f"{e.__class__.__module__}.{e.__class__.__name__}"


def load_original_model_and_inputs(
    model_original_src: str, context: dict
) -> tuple:
    """
    exec() the reference source. Returns (Model, get_init_inputs, get_inputs).
    """
    try:
        compile(model_original_src, "<string>", "exec")
    except SyntaxError as e:
        print(f"Syntax Error in original code {e}")
        return None

    try:
        exec(model_original_src, context)
    except Exception as e:
        print(f"Error in executing original code {e}")
        return None

    return (
        context.get("Model"),
        context.get("get_init_inputs"),
        context.get("get_inputs"),
    )


def load_custom_model(
    model_custom_src: str, context: dict, build_directory: str = None
) -> Optional[nn.Module]:
    """Load ModelNew via exec() (CUDA backend)."""
    if build_directory:
        context["BUILD_DIRECTORY"] = build_directory
        model_custom_src = (
            "import os\n"
            f"os.environ['TORCH_EXTENSIONS_DIR'] = '{build_directory}'\n"
        ) + model_custom_src

    try:
        compile(model_custom_src, "<string>", "exec")
        exec(model_custom_src, context)
    except SyntaxError as e:
        print(f"Syntax Error in custom generated code or Compilation Error {e}")
        return None

    return context.get("ModelNew")


def load_custom_model_with_tempfile(model_custom_src, entry_point="ModelNew"):
    """Load ModelNew via tempfile + importlib (Triton/TileLang/CuTe backend)."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as tmp_file:
        tmp_file.write(model_custom_src)
        tempfile_path = tmp_file.name
        temp_file = tmp_file

    spec = importlib.util.spec_from_file_location("temp_module", tempfile_path)
    temp_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(temp_module)

    ModelNew = getattr(temp_module, entry_point)
    return ModelNew, temp_file


def graceful_eval_cleanup(
    curr_context: dict,
    device: torch.device,
    temp_file=None,
):
    """Clean up GPU cache and optional tempfile after evaluation."""
    del curr_context
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device=device)
        torch.cuda.synchronize(device=device)
    if temp_file:
        temp_file.close()
        os.remove(temp_file.name)


def register_and_format_exception(
    exception_type: str,
    exception_msg,
    metadata: dict,
    verbose: bool = False,
    truncate=False,
    max_length=200,
):
    exception_str = str(exception_msg)
    if truncate and len(exception_str) > max_length:
        exception_str = exception_str[: max_length - 3] + "..."
    if verbose:
        print(f"[Exception {exception_type}] {exception_str}")
    metadata[exception_type] = exception_str
    return metadata


def run_and_check_correctness(
    original_model_instance: nn.Module,
    new_model_instance: nn.Module,
    get_inputs_fn: callable,
    metadata: dict,
    num_correct_trials: int,
    verbose: bool = False,
    seed: int = 42,
    device: Optional[torch.device] = None,
    backend: str = "cuda",
    precision: torch.dtype = torch.float32,
) -> KernelExecResult:
    """Run model and check correctness over multiple random-input trials."""
    pass_count = 0

    torch.manual_seed(seed)
    correctness_trial_seeds = [
        torch.randint(0, 2**32 - 1, (1,)).item()
        for _ in range(num_correct_trials)
    ]

    with torch.no_grad():
        for trial in range(num_correct_trials):
            trial_seed = correctness_trial_seeds[trial]
            if verbose:
                print(f"[Eval] Generating Random Input with seed {trial_seed}")

            set_seed(trial_seed)
            inputs = get_inputs_fn()
            inputs = [
                _process_input_tensor(x, device, backend, precision) for x in inputs
            ]

            set_seed(trial_seed)
            model = original_model_instance.to(device=device, dtype=precision)

            set_seed(trial_seed)
            model_new = new_model_instance.to(device=device, dtype=precision)

            output = model(*inputs)
            torch.cuda.synchronize(device=device)

            try:
                output_new = model_new(*inputs)
                torch.cuda.synchronize(device=device)

                def _flatten(x):
                    if isinstance(x, torch.Tensor):
                        return [x]
                    if isinstance(x, (tuple, list)):
                        flat = []
                        for item in x:
                            flat.extend(_flatten(item))
                        return flat
                    return []

                out_list = _flatten(output)
                out_new_list = _flatten(output_new)

                if len(out_list) != len(out_new_list):
                    metadata = register_and_format_exception(
                        "correctness_issue",
                        f"Output count mismatch: Expected {len(out_list)} tensors, got {len(out_new_list)}",
                        metadata,
                    )
                    metadata["correctness_issue_name"] = "correctness_issue"
                    return KernelExecResult(
                        compiled=True, correctness=False, metadata=metadata
                    )

                trial_passed = True
                tolerance = get_tolerance_for_precision(precision)
                fp8_dtypes = (getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None))
                fp8_dtypes = tuple(d for d in fp8_dtypes if d is not None)
                for oi, (o, o_new) in enumerate(zip(out_list, out_new_list)):
                    if o.dtype != o_new.dtype:
                        metadata = register_and_format_exception(
                            "correctness_issue",
                            f"Output[{oi}] dtype mismatch: Expected {o.dtype}, got {o_new.dtype}. "
                            f"The optimized kernel must produce outputs with the same dtype as the reference. "
                            f"Changing output types (e.g. fp8 -> float32) to absorb Python-side conversion "
                            f"overhead is not a valid optimization.",
                            metadata,
                        )
                        metadata["correctness_issue_name"] = "INTERFACE_CHANGED"
                        if verbose:
                            print(
                                f"[FAIL] trial {trial}: Output[{oi}] dtype mismatch: "
                                f"Expected {o.dtype}, got {o_new.dtype}"
                            )
                        return KernelExecResult(
                            compiled=True, correctness=False, metadata=metadata
                        )

                    if o.shape != o_new.shape:
                        metadata = register_and_format_exception(
                            "correctness_issue",
                            f"Output[{oi}] shape mismatch: Expected {o.shape}, got {o_new.shape}",
                            metadata,
                        )
                        metadata["correctness_issue_name"] = "correctness_issue"
                        if verbose:
                            print(
                                f"[FAIL] trial {trial}: Output[{oi}] shape mismatch: "
                                f"Expected {o.shape}, got {o_new.shape}"
                            )
                        return KernelExecResult(
                            compiled=True, correctness=False, metadata=metadata
                        )

                    if o.dtype in fp8_dtypes:
                        outputs_match = torch.equal(o, o_new)
                        max_diff = avg_diff = 0.0
                        if not outputs_match:
                            ou = o.view(torch.uint8)
                            on = o_new.view(torch.uint8)
                            max_diff = float((ou.to(torch.int32) - on.to(torch.int32)).abs().max().item())
                            avg_diff = float((ou.to(torch.int32) - on.to(torch.int32)).abs().float().mean().item())
                    else:
                        o_f = o.to(torch.float32)
                        o_new_f = o_new.to(torch.float32)
                        outputs_match = torch.allclose(
                            o_f, o_new_f, atol=tolerance, rtol=tolerance
                        )
                        max_diff = torch.max(torch.abs(o_f - o_new_f)).item()
                        avg_diff = torch.mean(torch.abs(o_f - o_new_f)).item()

                    if not outputs_match:
                        metadata.setdefault("max_difference", []).append(
                            f"{max_diff:.6f}"
                        )
                        metadata.setdefault("avg_difference", []).append(
                            f"{avg_diff:.6f}"
                        )
                        metadata["correctness_issue"] = f"Output[{oi}] mismatch"
                        if verbose:
                            print(f"[FAIL] trial {trial}: Output[{oi}] mismatch (max_diff={max_diff:.6f})")
                        trial_passed = False
                        break

                if trial_passed:
                    pass_count += 1
                    if verbose:
                        print(f"[PASS] trial {trial}: New Model matches Model")

            except Exception as e:
                print("[Error] Exception happens during correctness check")
                print(f"Error in launching kernel for ModelNew: {e}")
                print("\n[Full Traceback]:")
                traceback.print_exc()
                print()

                metadata = register_and_format_exception(
                    "runtime_error", e, metadata, truncate=True
                )
                metadata["runtime_error_name"] = get_error_name(e)
                metadata["runtime_error_traceback"] = traceback.format_exc()
                return KernelExecResult(
                    compiled=True, correctness=False, metadata=metadata
                )

    if verbose:
        print(
            f"[Eval] Pass count: {pass_count}, num_correct_trials: {num_correct_trials}"
        )

    metadata["correctness_trials"] = f"({pass_count} / {num_correct_trials})"

    if pass_count == num_correct_trials:
        return KernelExecResult(compiled=True, correctness=True, metadata=metadata)
    else:
        return KernelExecResult(compiled=True, correctness=False, metadata=metadata)


def eval_kernel_against_ref(
    original_model_src: str,
    custom_model_src: str,
    seed_num: int = 42,
    num_correct_trials: int = 1,
    num_perf_trials: int = 10,
    measure_performance: bool = False,
    timing_method: str = "cuda_event",
    verbose: bool = False,
    build_dir: os.PathLike = None,
    device: Union[torch.device, int] = None,
    backend: str = "cuda",
    precision: torch.dtype = torch.float32,
    check_for_excessive_speedup: bool = True,
    excessive_speedup_threshold: float = 10,
) -> KernelExecResult:
    """
    Evaluate a custom kernel against the reference model.

    Compiles and loads both models, checks correctness, and optionally
    measures performance (timing + speedup).
    """
    assert torch.cuda.is_available(), "CUDA is not available, cannot run Eval"

    if device is None:
        device = torch.cuda.current_device()

    torch.set_printoptions(
        precision=4, threshold=10, edgeitems=3, linewidth=80
    )

    torch.cuda.set_device(device)

    uses_tempfile = backend.lower() in ["triton", "tilelang", "cute"]

    metadata = {}
    metadata["hardware"] = torch.cuda.get_device_name(device=device)
    metadata["device"] = str(device)

    if uses_tempfile:
        if isinstance(device, int):
            device_num = device
        elif isinstance(device, torch.device):
            assert device.type == "cuda", "CUDA is not available on device"
            device_num = device.index
        else:
            raise ValueError(
                f"device must be an int or torch.device, got {type(device)}"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_num)

    context = {}

    if verbose:
        print(f"[Eval] Start Evaluation! on device: {device}")
        print("[Eval] Loading Original Model")

    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
        original_model_src, context
    )
    set_seed(seed_num)
    init_inputs = get_init_inputs()
    init_inputs = [
        _process_input_tensor(x, device, backend, precision) for x in init_inputs
    ]

    with torch.no_grad():
        set_seed(seed_num)
        original_model = Model(*init_inputs)
        assert hasattr(original_model, "forward")
        if verbose:
            print("[Eval] Original Model Loaded")

    if verbose:
        print("[Eval] Loading and Compiling New Model with Custom CUDA Kernel")

    # Compilation
    try:
        os.environ["TORCH_USE_CUDA_DSA"] = "1"
        temp_file = None

        if backend.lower() in ["triton", "tilelang", "cute"]:
            ModelNew, temp_file = load_custom_model_with_tempfile(
                custom_model_src, entry_point="ModelNew"
            )
        else:
            ModelNew = load_custom_model(custom_model_src, context, build_dir)
        torch.cuda.synchronize(device=device)
    except Exception as e:
        print(
            f"Failed to compile custom CUDA kernel: Record as compilation failure.\nError: {e}"
        )
        if "lock" in str(e) or "No such file or directory" in str(e):
            print(f"[Eval] Lock file error during compilation, Please retry. Error: {e}")
            graceful_eval_cleanup(context, device, temp_file)
            return None
        else:
            metadata["compilation_error_name"] = get_error_name(e)
            metadata["compilation_error"] = str(e)
            graceful_eval_cleanup(context, device, temp_file)
            return KernelExecResult(compiled=False, metadata=metadata)

    if ModelNew is None:
        print(
            "Failed to load custom model: Syntax error or ModelNew not found. "
            "Record as compilation failure."
        )
        metadata["compilation_error_name"] = "SyntaxError"
        metadata["compilation_error"] = (
            "Syntax error in custom generated code or ModelNew not found"
        )
        graceful_eval_cleanup(context, device, temp_file)
        return KernelExecResult(compiled=False, metadata=metadata)

    # Instantiate custom model
    try:
        with torch.no_grad():
            set_seed(seed_num)
            custom_model = ModelNew(*init_inputs)
            assert hasattr(custom_model, "forward")
            original_model = original_model.to(device=device, dtype=precision)
            custom_model = custom_model.to(device=device, dtype=precision)
            torch.cuda.synchronize(device=device)
        if verbose:
            print("[Eval] New Model with Custom CUDA Kernel Loaded")
    except RuntimeError as e:
        print(
            f"Failed to load custom CUDA kernel; Compiled but not able to run.\nError: {e}"
        )
        graceful_eval_cleanup(context, device, temp_file)
        metadata["runtime_error"] = str(e)
        metadata["runtime_error_name"] = get_error_name(e)
        return KernelExecResult(compiled=True, correctness=False, metadata=metadata)

    # Correctness
    kernel_exec_result = None
    if verbose:
        print("[Eval] Checking Correctness")
    try:
        kernel_exec_result = run_and_check_correctness(
            original_model,
            custom_model,
            get_inputs,
            metadata=metadata,
            num_correct_trials=num_correct_trials,
            verbose=verbose,
            seed=seed_num,
            device=device,
            backend=backend,
            precision=precision,
        )
    except Exception as e:
        metadata["runtime_error"] = str(e)
        metadata["runtime_error_name"] = get_error_name(e)
        kernel_exec_result = KernelExecResult(
            compiled=True, correctness=False, metadata=metadata
        )

    # Performance measurement
    if measure_performance:
        try:
            if kernel_exec_result and kernel_exec_result.correctness:
                if verbose:
                    print("[Eval] Measuring Performance as Sample is Correct")

                torch.cuda.synchronize(device=device)
                set_seed(seed_num)
                inputs = get_inputs()
                inputs = [
                    _process_input_tensor(x, device, backend, precision)
                    for x in inputs
                ]

                model_new = custom_model.to(device=device, dtype=precision)
                torch.cuda.synchronize(device=device)

                timing_fn = get_timing_function(timing_method)
                elapsed_times = timing_fn(
                    model_new,
                    inputs,
                    num_trials=num_perf_trials,
                    verbose=verbose,
                    device=device,
                )
                runtime_stats = get_timing_stats(elapsed_times, device=device)

                if verbose:
                    print(f"[Eval] Performance Stats: {runtime_stats}")
                kernel_exec_result.runtime = runtime_stats["mean"]
                kernel_exec_result.runtime_stats = runtime_stats

        except Exception as e:
            if verbose:
                print(f"[Eval] Error in Measuring Performance: {e}")
            kernel_exec_result.metadata["error_during_performance"] = str(e)

    # Reference timing (for speedup)
    if measure_performance and check_for_excessive_speedup:
        if verbose:
            print("[Eval] Additional checks to flag excessive speedup")

        torch.cuda.synchronize(device=device)
        set_seed(seed_num)
        inputs = get_inputs()
        inputs = [
            _process_input_tensor(x, device, backend, precision) for x in inputs
        ]

        torch.cuda.synchronize(device=device)

        timing_fn = get_timing_function(timing_method)
        reference_elapsed_times = timing_fn(
            original_model,
            inputs,
            num_trials=num_perf_trials,
            verbose=verbose,
            device=device,
        )
        reference_runtime_stats = get_timing_stats(
            reference_elapsed_times, device=device
        )
        kernel_exec_result.ref_runtime = reference_runtime_stats["mean"]
        kernel_exec_result.ref_runtime_stats = reference_runtime_stats

        effective_speedup = (
            kernel_exec_result.ref_runtime / kernel_exec_result.runtime
        )

        if verbose:
            print(
                f"[Eval] Effective Speedup is {effective_speedup:.2f}x "
                f"using timing method {timing_method}"
            )

        if effective_speedup > excessive_speedup_threshold:
            kernel_exec_result.metadata["excessive_speedup"] = True
            print(
                f"[WARNING] Excessive speedup {effective_speedup:.2f}x "
                f"over {excessive_speedup_threshold}x threshold detected"
            )
            print(
                "[WARNING] Double check your kernel carefully to ensure "
                "it is not reward hacking."
            )

    graceful_eval_cleanup(context, device, temp_file)
    return kernel_exec_result


###############################################################################
# Source transformation (from original kernelbench-bench.py)
###############################################################################


def read_file(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def rename_model_to_modelnew(src: str) -> str:
    """
    Rename `class Model(...)` -> `class ModelNew(...)` in solution source.
    Also renames `super(Model, self)` -> `super(ModelNew, self)`.
    """
    if re.search(r"\bclass\s+ModelNew\b", src):
        return src  # already has ModelNew

    src = re.sub(r"\bclass\s+Model\s*\(", "class ModelNew(", src)
    src = re.sub(r"\bsuper\s*\(\s*Model\s*,", "super(ModelNew,", src)
    return src


def _find_tail_section(src: str) -> int:
    """
    Find byte offset where the "tail section" begins (module-level code after
    the class body: variables like N=2048, get_inputs(), get_init_inputs()).
    """
    lines = src.split("\n")
    in_class = False

    for i, line in enumerate(lines):
        if re.match(r"^class\s+", line):
            in_class = True
            continue
        if in_class:
            if line.strip() == "" or (line and line[0] in (" ", "\t")):
                continue
            else:
                # First non-indented, non-blank line after class = tail
                return sum(len(l) + 1 for l in lines[:i])

    return len(src)


def prepare_solution_source(ref_src: str, sol_src: str) -> str:
    """
    Prepare solution source for eval:
    1. Rename class Model -> class ModelNew
    2. Strip solution's tail section (module vars, get_inputs, etc.)
    3. Append reference's tail section
    """
    modified = rename_model_to_modelnew(sol_src)

    sol_tail_start = _find_tail_section(modified)
    modified = modified[:sol_tail_start].rstrip()

    ref_tail_start = _find_tail_section(ref_src)
    ref_tail = ref_src[ref_tail_start:]

    if ref_tail.strip():
        modified = modified + "\n\n" + ref_tail
    else:
        modified = modified + "\n"

    return modified


###############################################################################
# Self-test for source transformation
###############################################################################


def _self_test():
    """Verify source transformation logic."""
    ref = '''import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return x * 2

N = 2048

def get_inputs():
    return [torch.randn(N, N)]

def get_init_inputs():
    return []
'''
    sol = '''import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return x * 3

N = 4096

def get_inputs():
    return [torch.randn(N)]

def get_init_inputs():
    return [42]
'''
    result = prepare_solution_source(ref, sol)

    # Must have ModelNew, not Model
    assert "class ModelNew(" in result, "ModelNew rename failed"
    assert "class Model(" not in result, "Original Model class still present"
    assert "super(ModelNew," in result, "super() rename failed"

    # Must have exactly one N = ... (from ref)
    n_count = len(re.findall(r"^N = ", result, re.MULTILINE))
    assert n_count == 1, f"Expected 1 N assignment, got {n_count}"
    assert "N = 2048" in result, "Should use ref's N = 2048"

    # Must have exactly one get_inputs and one get_init_inputs (from ref)
    gi_count = len(re.findall(r"^def get_inputs", result, re.MULTILINE))
    assert gi_count == 1, f"Expected 1 get_inputs, got {gi_count}"
    gii_count = len(re.findall(r"^def get_init_inputs", result, re.MULTILINE))
    assert gii_count == 1, f"Expected 1 get_init_inputs, got {gii_count}"

    # get_init_inputs should return [] (from ref, not [42] from sol)
    assert "return []" in result, "Should use ref's get_init_inputs"

    print("Self-test PASSED: source transformation is correct.")


###############################################################################
# Multi-shape support
###############################################################################


def _detect_shape_configs(ref_src: str) -> list[dict] | None:
    """Extract SHAPE_CONFIGS from reference source if defined."""
    ctx = {}
    try:
        exec(compile(ref_src, "<string>", "exec"), ctx)
    except Exception:
        return None
    configs = ctx.get("SHAPE_CONFIGS")
    if isinstance(configs, list) and len(configs) > 0:
        return configs
    return None


def _resolve_shape_indices(
    shape_configs: list[dict] | None, shape_idx: int | None
) -> list[int]:
    """Return list of shape indices to evaluate, or empty for legacy mode."""
    if shape_configs is None:
        return []
    if shape_idx is not None:
        if 0 <= shape_idx < len(shape_configs):
            return [shape_idx]
        raise ValueError(
            f"--shape-idx {shape_idx} out of range "
            f"(SHAPE_CONFIGS has {len(shape_configs)} entries)"
        )
    return list(range(len(shape_configs)))


def _inject_shape_idx(src: str, idx: int) -> str:
    """Patch get_inputs() calls to pass shape_idx=<idx> as default."""
    patched = re.sub(
        r"def get_inputs\(shape_idx=None\)",
        f"def get_inputs(shape_idx={idx})",
        src,
    )
    if patched == src:
        patched = re.sub(
            r"def get_inputs\(shape_idx=\d+\)",
            f"def get_inputs(shape_idx={idx})",
            src,
        )
    return patched


def _format_dims(cfg: dict) -> str:
    """Format shape config dims for display, skipping the label key."""
    parts = []
    for k, v in cfg.items():
        if k == "label":
            continue
        parts.append(f"{k}={v}")
    return ", ".join(parts)


def _run_single_eval(ref_src, sol_src, args, precision, baseline_src=None):
    """Legacy single-shape evaluation path."""
    result: KernelExecResult = eval_kernel_against_ref(
        original_model_src=ref_src,
        custom_model_src=sol_src,
        num_correct_trials=args.num_correct_trials,
        num_perf_trials=args.num_perf_trials,
        measure_performance=True,
        timing_method=args.timing_method,
        verbose=args.verbose,
        backend=args.backend,
        precision=precision,
    )

    if result is None:
        print("COMPILED: False")
        print("CORRECT: False")
        print("RUNTIME: -1")
        print("REF_RUNTIME: -1")
        print("SPEEDUP: -1")
        print("ERROR: eval_kernel_against_ref returned None (possible lock file error)")
        return None

    runtime_ms = result.runtime if result.runtime > 0 else -1
    ref_runtime_ms = result.ref_runtime if result.ref_runtime > 0 else -1
    speedup = (ref_runtime_ms / runtime_ms) if (runtime_ms > 0 and ref_runtime_ms > 0) else -1

    print(f"COMPILED: {result.compiled}")
    print(f"CORRECT: {result.correctness}")
    print(f"RUNTIME: {runtime_ms:.4f}" if runtime_ms > 0 else "RUNTIME: -1")
    print(f"REF_RUNTIME: {ref_runtime_ms:.4f}" if ref_runtime_ms > 0 else "REF_RUNTIME: -1")
    print(f"SPEEDUP: {speedup:.4f}x" if speedup > 0 else "SPEEDUP: -1")

    if baseline_src is not None and result.correctness and runtime_ms > 0:
        bl_ms = _measure_baseline_runtime(ref_src, baseline_src, args, precision)
        bl_speedup = (bl_ms / runtime_ms) if bl_ms > 0 else -1
        print(f"BASELINE_RUNTIME: {bl_ms:.4f}" if bl_ms > 0 else "BASELINE_RUNTIME: -1")
        print(f"BASELINE_SPEEDUP: {bl_speedup:.4f}x" if bl_speedup > 0 else "BASELINE_SPEEDUP: -1")

    if args.verbose:
        print()
        print("--- Details ---")
        if result.runtime_stats:
            print(f"Runtime stats: {result.runtime_stats}")
        if result.ref_runtime_stats:
            print(f"Ref runtime stats: {result.ref_runtime_stats}")
        if result.metadata:
            for k, v in result.metadata.items():
                print(f"  {k}: {v}")
    return result


def _print_multi_shape_summary(shape_results: list[dict], all_correct: bool):
    """Print aggregate summary after all shapes have been evaluated."""
    valid = [s for s in shape_results if s["runtime"] > 0 and s["ref_runtime"] > 0]
    print(f"SHAPES_EVALUATED: {len(shape_results)}")
    print(f"ALL_CORRECT: {all_correct}")

    for s in shape_results:
        tag = "PASS" if s["correct"] else "FAIL"
        rt = f'{s["runtime"]:.4f}' if s["runtime"] > 0 else "-1"
        sp = f'{s["speedup"]:.4f}x' if s["speedup"] > 0 else "-1"
        print(f"  [{tag}] {s['label']}: runtime={rt}ms speedup={sp}")

    if valid:
        avg_speedup = sum(s["speedup"] for s in valid) / len(valid)
        min_speedup = min(s["speedup"] for s in valid)
        max_speedup = max(s["speedup"] for s in valid)
        print(f"AVG_SPEEDUP: {avg_speedup:.4f}x")
        print(f"MIN_SPEEDUP: {min_speedup:.4f}x")
        print(f"MAX_SPEEDUP: {max_speedup:.4f}x")

    worst = min(
        (s for s in shape_results if s["correct"]),
        key=lambda s: s.get("speedup", float("inf")),
        default=None,
    )
    if worst and worst["speedup"] > 0:
        print(f"SPEEDUP: {worst['speedup']:.4f}x")
        print(f"RUNTIME: {worst['runtime']:.4f}")
        print(f"REF_RUNTIME: {worst['ref_runtime']:.4f}")

    bl_valid = [s for s in shape_results if s.get("baseline_speedup", -1) > 0]
    if bl_valid:
        worst_bl = min(bl_valid, key=lambda s: s["baseline_speedup"])
        print(f"BASELINE_SPEEDUP: {worst_bl['baseline_speedup']:.4f}x")

    print(f"COMPILED: True")
    print(f"CORRECT: {all_correct}")


###############################################################################
# Baseline comparison
###############################################################################


def _measure_baseline_runtime(
    ref_src: str,
    baseline_src: str,
    args,
    precision: torch.dtype,
) -> float:
    """Time the baseline kernel and return mean runtime in ms, or -1."""
    modified_bl = prepare_solution_source(ref_src, baseline_src)
    bl_result = eval_kernel_against_ref(
        original_model_src=ref_src,
        custom_model_src=modified_bl,
        num_correct_trials=1,
        num_perf_trials=args.num_perf_trials,
        measure_performance=True,
        timing_method=args.timing_method,
        verbose=False,
        backend=args.backend,
        precision=precision,
        check_for_excessive_speedup=False,
    )
    if bl_result and bl_result.runtime > 0:
        return bl_result.runtime
    return -1.0


###############################################################################
# CLI
###############################################################################


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate optimized kernel against reference (self-contained KernelBench)"
    )
    parser.add_argument(
        "--ref", required=True, help="Path to reference kernel (with class Model)"
    )
    parser.add_argument(
        "--solution",
        required=True,
        help="Path to optimized kernel (with class Model - renamed automatically)",
    )
    parser.add_argument(
        "--timing-method",
        default="cuda_event",
        choices=["cuda_event", "host_time"],
        help="GPU timing method (default: cuda_event)",
    )
    parser.add_argument(
        "--precision",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Precision for evaluation (default: float32)",
    )
    parser.add_argument(
        "--backend",
        default="cuda",
        choices=["cuda", "triton", "tilelang", "cute", "hip"],
        help="Backend for kernel compilation (default: cuda)",
    )
    parser.add_argument(
        "--num-correct-trials",
        type=int,
        default=5,
        help="Number of correctness trials (default: 5)",
    )
    parser.add_argument(
        "--num-perf-trials",
        type=int,
        default=100,
        help="Number of performance trials (default: 100)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--shape-idx",
        type=int,
        default=None,
        help="Run only one shape from SHAPE_CONFIGS (0-based index). "
        "If omitted and SHAPE_CONFIGS exists, all shapes are evaluated.",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Path to baseline kernel. When provided, outputs BASELINE_RUNTIME "
        "and BASELINE_SPEEDUP (= baseline_runtime / solution_runtime).",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run source transformation self-test and exit",
    )

    args = parser.parse_args()

    if args.self_test:
        _self_test()
        sys.exit(0)

    precision_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    precision = precision_map[args.precision]

    ref_src = read_file(args.ref)
    sol_src = read_file(args.solution)
    baseline_src = read_file(args.baseline) if args.baseline else None

    modified_sol_src = prepare_solution_source(ref_src, sol_src)

    if args.verbose:
        print("=" * 60)
        print("REFERENCE SOURCE:")
        print("=" * 60)
        print(ref_src[:500], "..." if len(ref_src) > 500 else "")
        print()
        print("=" * 60)
        print("MODIFIED SOLUTION SOURCE:")
        print("=" * 60)
        print(modified_sol_src[:500], "..." if len(modified_sol_src) > 500 else "")
        print()

    shape_configs = _detect_shape_configs(ref_src)
    shape_indices = _resolve_shape_indices(shape_configs, args.shape_idx)

    if not shape_indices:
        result = _run_single_eval(
            ref_src, modified_sol_src, args, precision, baseline_src
        )
        sys.exit(0 if (result and result.correctness) else 1)

    all_correct = True
    shape_results = []

    for idx in shape_indices:
        cfg = shape_configs[idx]
        label = cfg.get("label", f"shape_{idx}")
        print()
        print(f"{'=' * 60}")
        print(f"SHAPE [{idx}]: {label}  dims={_format_dims(cfg)}")
        print(f"{'=' * 60}")

        patched_ref = _inject_shape_idx(ref_src, idx)
        patched_sol = _inject_shape_idx(modified_sol_src, idx)

        result = eval_kernel_against_ref(
            original_model_src=patched_ref,
            custom_model_src=patched_sol,
            num_correct_trials=args.num_correct_trials,
            num_perf_trials=args.num_perf_trials,
            measure_performance=True,
            timing_method=args.timing_method,
            verbose=args.verbose,
            backend=args.backend,
            precision=precision,
        )

        if result is None:
            print(f"SHAPE_{idx}_LABEL: {label}")
            print(f"SHAPE_{idx}_COMPILED: False")
            print(f"SHAPE_{idx}_CORRECT: False")
            print(f"SHAPE_{idx}_RUNTIME: -1")
            print(f"SHAPE_{idx}_REF_RUNTIME: -1")
            print(f"SHAPE_{idx}_SPEEDUP: -1")
            all_correct = False
            shape_results.append({
                "idx": idx, "label": label,
                "compiled": False, "correct": False,
                "runtime": -1, "ref_runtime": -1, "speedup": -1,
            })
            continue

        runtime_ms = result.runtime if result.runtime > 0 else -1
        ref_runtime_ms = result.ref_runtime if result.ref_runtime > 0 else -1
        speedup = (ref_runtime_ms / runtime_ms) if (runtime_ms > 0 and ref_runtime_ms > 0) else -1

        print(f"SHAPE_{idx}_LABEL: {label}")
        print(f"SHAPE_{idx}_COMPILED: {result.compiled}")
        print(f"SHAPE_{idx}_CORRECT: {result.correctness}")
        print(f"SHAPE_{idx}_RUNTIME: {runtime_ms:.4f}" if runtime_ms > 0 else f"SHAPE_{idx}_RUNTIME: -1")
        print(f"SHAPE_{idx}_REF_RUNTIME: {ref_runtime_ms:.4f}" if ref_runtime_ms > 0 else f"SHAPE_{idx}_REF_RUNTIME: -1")
        print(f"SHAPE_{idx}_SPEEDUP: {speedup:.4f}x" if speedup > 0 else f"SHAPE_{idx}_SPEEDUP: -1")

        if not result.correctness:
            all_correct = False

        shape_results.append({
            "idx": idx, "label": label,
            "compiled": result.compiled, "correct": result.correctness,
            "runtime": runtime_ms, "ref_runtime": ref_runtime_ms, "speedup": speedup,
        })

        if args.verbose and result.runtime_stats:
            print(f"  Runtime stats: {result.runtime_stats}")
        if args.verbose and result.ref_runtime_stats:
            print(f"  Ref runtime stats: {result.ref_runtime_stats}")

        if baseline_src is not None and result.correctness and runtime_ms > 0:
            patched_bl = _inject_shape_idx(
                prepare_solution_source(patched_ref, baseline_src), idx
            )
            bl_result = eval_kernel_against_ref(
                original_model_src=patched_ref,
                custom_model_src=patched_bl,
                num_correct_trials=1,
                num_perf_trials=args.num_perf_trials,
                measure_performance=True,
                timing_method=args.timing_method,
                verbose=False,
                backend=args.backend,
                precision=precision,
                check_for_excessive_speedup=False,
            )
            bl_ms = bl_result.runtime if (bl_result and bl_result.runtime > 0) else -1
            bl_speedup = (bl_ms / runtime_ms) if bl_ms > 0 else -1
            print(f"SHAPE_{idx}_BASELINE_RUNTIME: {bl_ms:.4f}" if bl_ms > 0 else f"SHAPE_{idx}_BASELINE_RUNTIME: -1")
            print(f"SHAPE_{idx}_BASELINE_SPEEDUP: {bl_speedup:.4f}x" if bl_speedup > 0 else f"SHAPE_{idx}_BASELINE_SPEEDUP: -1")
            shape_results[-1]["baseline_speedup"] = bl_speedup

    print()
    print("=" * 60)
    print("MULTI-SHAPE SUMMARY")
    print("=" * 60)
    _print_multi_shape_summary(shape_results, all_correct)

    sys.exit(0 if all_correct else 1)


if __name__ == "__main__":
    main()
