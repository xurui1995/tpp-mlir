# ===============================================================================
# Copyright 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================
from time import sleep
from tpp import ir
from tpp.passmanager import *
from tpp.execution_engine import *
from utils import *
from timeit import timeit, repeat
from enhance import *
import os
import random


def lowerToLLVM(module: ir.Module, passes: str, ctx: ir.Context, ir_printing=False):
    pm = PassManager.parse(passes, ctx)
    if ir_printing:
        pm.enable_ir_printing()
    pm.run(module.operation)
    return module


def python_bench(
    ctx: ir.Context,
    ir_module: ir.Module,
    entry_name: str,
    passes: str,
    mlir_args: list,
    ir_printing=False,
    repeat_time=100,
    warm_up=20,
) -> float:
    ## option 1 (python timeit)
    print("python timeit")
    engine = ExecutionEngine(
        lowerToLLVM(ir_module, passes, ctx, ir_printing),
        opt_level=3,
        shared_libs=[
            os.getenv("MLIR_C_RUNNER_UTILS", ""),
            os.getenv("MLIR_RUNNER_UTILS", ""),
            "/home/xurui/mlir/tpp-mlir/build/lib/libtpp_xsmm_runner_utils.so"
        ],
    )

    func = engine.lookup(entry_name)
    packed_args = (ctypes.c_void_p * len(mlir_args))()
    for argNum in range(len(mlir_args)):
        packed_args[argNum] = ctypes.cast(mlir_args[argNum], ctypes.c_void_p)

    def run_bench(func, arg):
        func(arg)

    timeit(lambda: run_bench(func, packed_args), number=warm_up)
    total_time = timeit(lambda: run_bench(func, packed_args), number=repeat_time)
    print(total_time * 1000 / repeat_time, "ms")
    return total_time * 1000 / repeat_time


def MBR_bench(
    ctx: ir.Context,
    ir_module: ir.Module,
    entry_name: str,
    passes: str,
    mlir_args: list,
    ir_printing=False,
    repeat_time=100,
    warm_up=20,
) -> float:
    kernel_func = get_kernel_func_from_module(ir_module, entry_name)
    timer_func = emit_timer_func()
    wrapped_func = emit_benchmark_wrapped_main_func(kernel_func, timer_func)
    main_module_with_benchmark = ir.Module.parse(
        str(timer_func) + str(wrapped_func) + str(kernel_func)
    )
    # print(str(main_module_with_benchmark))
    engine = ExecutionEngine(
        lowerToLLVM(main_module_with_benchmark, passes, ctx, ir_printing),
        opt_level=3,
        shared_libs=[
            os.getenv("MLIR_C_RUNNER_UTILS", ""),
            os.getenv("MLIR_RUNNER_UTILS", ""),
            "/home/xurui/mlir/tpp-mlir/build/lib/libtpp_xsmm_runner_utils.so"
        ],
    )
    np_timers_ns = np.array([0], dtype=np.int64)
    arg2_memref_ptr = ctypes.pointer(
        ctypes.pointer(get_ranked_memref_descriptor(np_timers_ns))
    )
    total_time = 0
    ns_to_ms_scale = 1e-6

    def run(engine_invoke, bench_func_name, *mlir_args):
        engine_invoke(bench_func_name, *mlir_args)

    for i in range(repeat_time + warm_up):
        run(engine.invoke, "main", *mlir_args, arg2_memref_ptr)
        if i >= warm_up:
            total_time += int(np_timers_ns[0]) * ns_to_ms_scale

    print(total_time / repeat_time, "ms")
    return total_time / repeat_time


def fake_bench() -> float:
    sleep(1)
    return float(random.randint(1, 100))
