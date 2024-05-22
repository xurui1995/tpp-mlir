
// module {
//   func.func @main_entry(%arg0: tensor<10x10xbf16>, %arg1: tensor<10x10xbf16>, %arg2: tensor<10xbf16>) -> tensor<10x10xbf16> attributes {llvm.emit_c_interface} {
//     %0 = onednn_graph.matmul %arg0, %arg1, %arg2 {KBlock = 64 : i32} : (tensor<10x10xbf16>, tensor<10x10xbf16>, tensor<10xbf16>) -> tensor<10x10xbf16>
//     return %0 : tensor<10x10xbf16>
//   }
// }

// module {
//   func.func @main_entry(%arg0: tensor<10x10xbf16>, %arg1: tensor<10x10xbf16>, %arg2: tensor<10xbf16>) -> tensor<10x10xbf16> attributes {llvm.emit_c_interface} {
//     %0 = onednn_graph.matmul %arg0, %arg1, %arg2 : (tensor<10x10xbf16>, tensor<10x10xbf16>, tensor<10xbf16>) -> tensor<10x10xbf16>
//     return %0 : tensor<10x10xbf16>
//   }
// }

// module {
//   func.func @main_entry(%arg0: tensor<32x32x32x32xbf16>, %arg1: tensor<32x32x32x32xbf16>) -> tensor<32x32x32x32xbf16> attributes {llvm.emit_c_interface} {
//     %0 = tensor.empty() : tensor<32x32x32x32xbf16>
//     %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<32x32x32x32xbf16>, tensor<32x32x32x32xbf16>) outs(%0 : tensor<32x32x32x32xbf16>) {
//     ^bb0(%in: bf16, %in_1: bf16, %out: bf16):
//       %3 = arith.mulf %in, %in_1 : bf16
//       %4 = arith.addf %out, %3 : bf16
//       linalg.yield %4 : bf16
//     } -> tensor<32x32x32x32xbf16>
//     return %1 : tensor<32x32x32x32xbf16>
//   }
// }

// module {
//   func.func @main_entry(%arg0: tensor<32x32x32x32xbf16>, %arg1: tensor<32x32x32x32xbf16>) -> tensor<32x32x32x32xbf16> attributes {llvm.emit_c_interface} {
//     %0 = tensor.empty() : tensor<32x32x32x32xbf16>
//     %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<32x32x32x32xbf16>, tensor<32x32x32x32xbf16>) outs(%0 : tensor<32x32x32x32xbf16>) {
//     ^bb0(%in: bf16, %in_1: bf16, %out: bf16):
//       %3 = arith.mulf %in, %in_1 : bf16
//       %4 = arith.addf %out, %3 : bf16
//       linalg.yield %4 : bf16
//     } -> tensor<32x32x32x32xbf16>
//     return %1 : tensor<32x32x32x32xbf16>
//   }
// }


// module {
//   func.func @main_entry(%arg0: tensor<512x512xbf16>, %arg1: tensor<512x512xbf16>) -> tensor<512x512xbf16> attributes {llvm.emit_c_interface} {
//     %0 = tensor.empty() : tensor<512x512xbf16>
//     %cst = arith.constant 2.000000e+00 : bf16
//     %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<512x512xbf16>) -> tensor<512x512xbf16>
//     %2 = linalg.fill ins(%cst : bf16) outs(%arg0 : tensor<512x512xbf16>) -> tensor<512x512xbf16>
//     %3 = linalg.fill ins(%cst : bf16) outs(%arg1 : tensor<512x512xbf16>) -> tensor<512x512xbf16>
//     %4 = linalg.matmul ins(%2, %3 : tensor<512x512xbf16>, tensor<512x512xbf16>) outs(%1 : tensor<512x512xbf16>) -> tensor<512x512xbf16> 
//     return %4 : tensor<512x512xbf16>
//   }
// }
