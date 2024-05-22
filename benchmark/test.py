import gc, sys, os, tempfile
from tpp.ir import *
from tpp.passmanager import *
from tpp.execution_engine import *
from tpp.runtime import *


def log(*args):
    print(*args, file=sys.stderr)
    sys.stderr.flush()


def run(f):
    log("\nTEST:", f.__name__)
    f()
    gc.collect()
    assert Context._get_live_count() == 0


def lowerToLLVM(module):
    pm = PassManager.parse(
        """
            builtin.module(  
                convert-openmp-to-llvm
                convert-complex-to-llvm,
                finalize-memref-to-llvm,
                convert-func-to-llvm,
                reconcile-unrealized-casts)
        """
    )
    pm.run(module.operation)
    return module


# def testInvokeFloatMax():
#     with Context():
#         module = Module.parse(
#             r"""
#             func.func @max(%arg0: f32, %arg1: f32) -> f32 attributes { llvm.emit_c_interface } {
#                 %add = arith.maximumf %arg0, %arg1 : f32
#                 return %add : f32
#             }
#             """
#         )
#         execution_engine = ExecutionEngine(lowerToLLVM(module))
#         c_float_p = ctypes.c_float * 1
#         arg0 = c_float_p(1.0)
#         arg1 = c_float_p(2.0)
#         res = c_float_p(-1.0)
#         execution_engine.invoke("max", arg0, arg1, res)
#         print(f"max({arg0[0]},{arg1[0]}) = {res[0]}")


# run(testInvokeFloatMax)


def testSharedLibLoad():
    with Context():
        module = Module.parse(
            """
            module {
                func.func @main(): {llvm.emit_c_interface} {
                    %1 = alloc() : memref<100000xi32>    
                    %i0 = constant 0 : index
                    %i100 = constant 100 : index
                    %i1 = constant 1 : index
                    %i2 = constant 2 : index

                    omp.parallel{
                    scf.for %arg0 = %i0 to %i100 step %i1{
                        %arg0_cast = index_cast %arg0 : index to i32 
                        
                    } 
                    }
                    return
                }
                }
            """
        )
        # arg0 = np.array([0.0]).astype(np.float32)

        # arg0_memref_ptr = ctypes.pointer(
        #     ctypes.pointer(get_ranked_memref_descriptor(arg0))
        # )
        shared_libs = [
            "/home/shared/bin/mlir/20240401/lib/libmlir_c_runner_utils.so",
            "/home/shared/bin/mlir/20240401/lib/libmlir_runner_utils.so",
        ]
        execution_engine = ExecutionEngine(
            lowerToLLVM(module), opt_level=3, shared_libs=shared_libs
        )
        execution_engine.invoke("main")
        # CHECK: Unranked Memref
        # CHECK-NEXT: [42]


def testSharedLibLoad_2():

    #    %1 = tensor.empty() : tensor<2x2xf32>
    #                 %cst = arith.constant 0.000000e+00 : f32
    #                 %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<2x2xf32>) -> tensor<2x2xf32>
    #                 %3 = linalg.matmul ins(%arg1, %arg2 : tensor<2x2xf32>, tensor<2x2xf32>) outs(%2 : tensor<2x2xf32>) -> tensor<2x2xf32>
    #                 return %3 : tensor<2x2xf32>

    #  %1 = linalg.fill ins(%cst42 : f32) outs(%arg0 : tensor<2x2xf32>) -> tensor<2x2xf32>
    #             %pos = arith.constant 0 : index
    #             %2 = tensor.extract %1[%pos, %pos] : tensor<2x2xf32>
    #             return %2 : f32
    with Context() as ctx:
        # ctx.enable_multithreading(False)
        module = Module.parse(
            """
module {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func public @main_entry(%arg0: tensor<512x512xf32>, %arg1: tensor<512x512xf32>, %arg2: memref<?xi64>) attributes {llvm.emit_c_interface} {
    %0 = call @nanoTime() : () -> i64
    %1 = call @ff(%arg0, %arg1) : (tensor<512x512xf32>, tensor<512x512xf32>) -> tensor<512x512xf32>
    %2 = call @nanoTime() : () -> i64
    %3 = arith.subi %2, %0 : i64
    %c0 = arith.constant 0 : index
    memref.store %3, %arg2[%c0] : memref<?xi64>
    return
  }
  func.func @ff(%arg0: tensor<512x512xf32>, %arg1: tensor<512x512xf32>) -> tensor<512x512xf32> attributes {llvm.emit_c_interface} {
    %0 = tensor.empty() : tensor<512x512xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<512x512xf32>) -> tensor<512x512xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<512x512xf32>, tensor<512x512xf32>) outs(%1 : tensor<512x512xf32>) -> tensor<512x512xf32>
    return %2 : tensor<512x512xf32>
  }
}
            """
        )
        arg0 = np.ones([512,512], np.float32)
        arg0_memref_ptr = ctypes.pointer(
            ctypes.pointer(get_ranked_memref_descriptor(arg0))
        )

        arg1 = np.ones([512,512], np.float32)
        arg1_memref_ptr = ctypes.pointer(
            ctypes.pointer(get_ranked_memref_descriptor(arg1))
        )
        
        np_timers_ns = np.array([0], dtype=np.int64)
        
        arg2_memref_ptr =  ctypes.pointer(
                ctypes.pointer(get_ranked_memref_descriptor(np_timers_ns))
            )
        

        # ref_out = make_nd_memref_descriptor(2, ctypes.c_float)()
        # mem_out = ctypes.pointer(ctypes.pointer(ref_out))

        # arg2 = np.array([[0.0, 0.0], [0.0, 0.0]]).astype(np.float32)
        # arg2_memref_ptr = ctypes.pointer(
        #     ctypes.pointer(get_ranked_memref_descriptor(arg2))
        # )

        shared_libs = [
            "/home/shared/bin/mlir/20240401/lib/libmlir_c_runner_utils.so",
            "/home/shared/bin/mlir/20240401/lib/libmlir_runner_utils.so",
        ]

        def lower(module):
            pm = PassManager.parse(
                """
        builtin.module(    
            func.func(linalg-generalize-named-ops),
            func.func(linalg-fuse-elementwise-ops),
            convert-shape-to-std,
            one-shot-bufferize,
            cse,
            func-bufferize,
            func.func(bufferization-bufferize),
            func.func(finalizing-bufferize),
            func.func(buffer-deallocation-pipeline),
            func.func(convert-linalg-to-parallel-loops),
            func.func(lower-affine),  
            convert-scf-to-cf, 
            func.func(arith-expand),
            func.func(convert-math-to-llvm),
            convert-math-to-libm,expand-strided-metadata,finalize-memref-to-llvm,lower-affine,convert-bufferization-to-memref,finalize-memref-to-llvm,
            func.func(convert-arith-to-llvm),
            convert-func-to-llvm,
            convert-cf-to-llvm,convert-complex-to-llvm,reconcile-unrealized-casts
        )
                """
            )
            # pm.enable_ir_printing()
            pm.run(module.operation)
            return module

        execution_engine = ExecutionEngine(
            lower(module), opt_level=3, shared_libs=shared_libs
        )

        # arg00 = np.array([0.0]).astype(np.float32)
        # arg00_memref_ptr = ctypes.pointer(
        #     ctypes.pointer(get_ranked_memref_descriptor(arg00))
        # )

        c_float_p = ctypes.c_float * 1
        res = c_float_p(-1.0)
        
        mlir_args = [arg0_memref_ptr, arg1_memref_ptr, arg2_memref_ptr]
        func = execution_engine.lookup("main_entry")
        packed_args = (ctypes.c_void_p * len(mlir_args))()
        for argNum in range(len(mlir_args)):
            packed_args[argNum] = ctypes.cast(mlir_args[argNum], ctypes.c_void_p)
        func(packed_args)
        
        # execution_engine.invoke("main_entry", arg0_memref_ptr, arg1_memref_ptr, arg2_memref_ptr)
        print(int(np_timers_ns[0]))
        # print(type(arg0))

        # print("arg0", ranked_memref_to_numpy(mem_out[0]))
        # print("res", res[0])


run(testSharedLibLoad)
