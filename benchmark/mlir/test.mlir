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