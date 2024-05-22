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
from copy import deepcopy
import ctypes
from enhance import *
import numpy as np
from tpp import ir
from tpp.dialects import func, arith, memref
from typing import Callable
from op_config import *
import ml_dtypes


def emit_timer_func() -> func.FuncOp:
    """Returns the declaration of nanoTime function. If nanoTime function is
    used, the `MLIR_RUNNER_UTILS` and `MLIR_C_RUNNER_UTILS` must be included.
    """
    i64_type = ir.IntegerType.get_signless(64)
    nanoTime = func.FuncOp("nanoTime", ([], [i64_type]), visibility="private")
    nanoTime.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
    return nanoTime


def emit_benchmark_wrapped_main_func(kernel_func, timer_func):
    i64_type = ir.IntegerType.get_signless(64)
    memref_of_i64_type = ir.MemRefType.get([1], i64_type)
    wrapped_func = func.FuncOp(
        # Same signature and an extra buffer of indices to save timings.
        "main",
        (kernel_func.arguments.types + [memref_of_i64_type], kernel_func.type.results),
        visibility="public",
    )
    wrapped_func.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
    with ir.InsertionPoint(wrapped_func.add_entry_block()):
        timer_buffer = wrapped_func.arguments[-1]
        start = func.CallOp(timer_func, [])
        call_op = func.CallOp(
            kernel_func,
            list(wrapped_func.arguments[:-1]),
        )
        end = func.CallOp(timer_func, [])
        time_taken = arith.SubIOp(end, start)
        zero = arith.ConstantOp.create_index(0)
        memref.StoreOp(time_taken, timer_buffer, [zero])
        func.ReturnOp(call_op.results)
    return wrapped_func


def numpy_to_ctypes(np_dtype):
    if np_dtype == np.int32:
        return ctypes.c_int
    elif np_dtype == np.float64:
        return ctypes.c_double
    elif np_dtype == np.uint8:
        return ctypes.c_ubyte
    elif np_dtype == np.int8:
        return ctypes.c_byte
    elif np_dtype == np.uint16:
        return ctypes.c_ushort
    elif np_dtype == np.int16:
        return ctypes.c_short
    elif np_dtype == np.uint32:
        return ctypes.c_uint
    elif np_dtype == np.int64:
        return ctypes.c_longlong
    elif np_dtype == np.uint64:
        return ctypes.c_ulonglong
    elif np_dtype == np.float32:
        return ctypes.c_float
    elif np_dtype == ml_dtypes.bfloat16:
        return ctypes.c_int16
    else:
        raise ValueError("Unsupported NumPy data type")


def np_args_to_mlir_args(np_args: list[np.ndarray]) -> list:
    mlir_args = []
    for arg in np_args:
        mlir_args.append(
            ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(arg)))
        )
    return mlir_args


def np_res_to_mlir_res(np_res: list[np.ndarray]) -> list:
    mlir_res = []
    for res in np_res:
        print(res.dtype, res.dtype == ml_dtypes.bfloat16)
        mlir_res.append(
            ctypes.pointer(
                ctypes.pointer(
                    make_nd_memref_descriptor(res.ndim, numpy_to_ctypes(res.dtype))()
                )
            )
        )
    return mlir_res


def get_tensor_mlir_args(f: func.FuncOp, np_args: list[np.ndarray]):
    compiled_program_args = []
    for res in np_args[: len(f.type.results)]:
        compiled_program_args.append(
            ctypes.pointer(
                ctypes.pointer(
                    make_nd_memref_descriptor(res.ndim, numpy_to_ctypes(res.dtype))()
                )
            )
        )
    for arg in np_args[len(f.type.results) :]:
        compiled_program_args.append(
            ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(arg)))
        )
    return compiled_program_args


def mlir_type(s, ctx):
    type_mapping = {
        "f32": ir.F32Type.get(ctx),
        "f64": ir.F64Type.get(ctx),
        "bf16": ir.BF16Type.get(ctx),
        "i32": ir.IntegerType.get_signed(32),
        "i8": ir.IntegerType.get_signed(8),
    }
    return type_mapping[s]


def make_tensor(tensor_type):
    type_mapping = {
        "f32": np.float32,
        "bf16": ml_dtypes.bfloat16,
        "f64": np.float64,
        "i8": np.int8,
        "i16": np.int16,
        "i32": np.int32,
        "i64": np.int64,
        "u8": np.uint8,
        "u16": np.uint16,
        "u32": np.uint32,
        "u64": np.uint64,
    }
    return np.ones(tensor_type.shape, type_mapping[str(tensor_type.element_type)])


def get_kernel_func_from_module(
    module: ir.Module, func_name: str = "main_entry"
) -> func.FuncOp:
    assert (
        len(module.operation.regions) == 1
    ), "Expected kernel module to have only one region"
    assert (
        len(module.operation.regions[0].blocks) == 1
    ), "Expected kernel module to have only one block"
    for f in module.operation.regions[0].blocks[0].operations:
        if type(f) is func.FuncOp and str(f.name).strip('"') == func_name:
            return f
    raise ValueError("can not find the entry function")


def get_default_passes():
    passes = """
builtin.module(default-tpp-passes{linalg-to-loops=false parallel-task-grid=2,8},expand-strided-metadata,func.func(convert-perf-to-loops),convert-perf-to-func,convert-tensor-to-linalg,func.func(convert-linalg-to-loops),convert-scf-to-openmp{num-threads=0},convert-vector-to-scf{full-unroll=false lower-tensors=false target-rank=1},arith-expand{include-bf16=false},lower-affine,print-ir{label=},convert-vector-to-llvm{enable-amx=false enable-arm-neon=false enable-arm-sve=false enable-x86vector=false force-32bit-vector-indices=true reassociate-fp-reductions=false},finalize-memref-to-llvm{index-bitwidth=0 use-aligned-alloc=false use-generic-functions=false},convert-scf-to-cf,convert-openmp-to-llvm,convert-math-to-llvm{approximate-log1p=true},func.func(gpu-async-region),gpu-to-llvm{gpu-binary-annotation=gpu.binary use-bare-pointers-for-host=false use-bare-pointers-for-kernels=false},gpu-module-to-binary{format=fatbin  opts= toolkit=},async-to-async-runtime,async-runtime-ref-counting,convert-async-to-llvm,convert-func-to-llvm{index-bitwidth=0 use-bare-ptr-memref-call-conv=false},func.func(convert-arith-to-llvm{index-bitwidth=0}),func.func(canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=true test-convergence=false top-down=true}),func.func(cse),reconcile-unrealized-casts,launch-func-to-vulkan,symbol-dce)

    """

    # convert-scf-to-openmp,
    # convert-openmp-to-llvm,
    # -convert-index-to-llvm
    return passes


def to_int_vector(s: str) -> list[int]:
    if not s or len(s) == 0:
        return []
    return [int(i) for i in s.strip().split("x")]


def to_bool_vector(s: str) -> list[bool]:
    if not s or len(s) == 0:
        return []
    return [bool(i) for i in s.strip().split("x")]


def walk_operations(op: ir.Operation, callback=None):
    for region in op.regions:
        for block in region:
            for child_op in block:
                if callback:
                    callback(child_op)
                walk_operations(child_op, callback)


def get_all_tunable_ops(op: ir.Operation):
    tunable_ops = []
    for region in op.regions:
        for block in region:
            for child_op in block:
                if child_op.name == "onednn_graph.matmul":
                    tunable_ops.append(child_op)
                tunable_ops = tunable_ops + get_all_tunable_ops(child_op)
    return tunable_ops


def walk_and_print_operations(op, indent=""):
    for i, region in enumerate(op.regions):
        print(f"{indent}REGION {i}:")
        for j, block in enumerate(region):
            print(f"{indent}  BLOCK {j}:")
            for k, child_op in enumerate(block):
                print(f"{indent}    OP {k}: {child_op}")
                walk_and_print_operations(indent + "      ", child_op)


def attach_configs_to_ir(ir_module: ir.Module, configs: list[Config]):
    # ctx.allow_unregistered_dialects=True
    tunable_ops = get_all_tunable_ops(ir_module.operation)
    assert len(tunable_ops) == len(
        configs
    ), "tunable ops and configs should have the same length"
    for i, op in enumerate(tunable_ops):
        if op.name == "onednn_graph.matmul":
            op.attributes["MBlock"] = ir.IntegerAttr.get(
                ir.IntegerType.get_signless(32), configs[i].M_block
            )
            op.attributes["KBlock"] = ir.IntegerAttr.get(
                ir.IntegerType.get_signless(32), configs[i].K_block
            )
            op.attributes["NBlock"] = ir.IntegerAttr.get(
                ir.IntegerType.get_signless(32), configs[i].N_block
            )


def extract_configs_from_ir(ir_module: ir.Module):
    tunable_ops = get_all_tunable_ops(ir_module.operation)
    configs = []
    for op in tunable_ops:
        if op.name == "onednn_graph.matmul":
            cfg = MatMulConfig()
            if "MBlock" in op.attributes:
                cfg.M_block = op.attributes["MBlock"].value
            if "NBlock" in op.attributes:
                cfg.N_block = op.attributes["NBlock"].value
            if "KBlock" in op.attributes:
                cfg.K_block = op.attributes["KBlock"].value
            configs.append(cfg)
    return configs


def load_mlir_from_path(path: str) -> str:
    with open(path, "r") as file:
        content = file.read()
    return content


def register_gc_dialects():
    pass
