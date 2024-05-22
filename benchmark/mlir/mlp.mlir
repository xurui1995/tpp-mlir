module {
  func.func @main_entry(%arg0: tensor<128x13xf32>, %arg1: tensor<13x512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512x256xf32>, %arg4: tensor<256xf32>, %arg5: tensor<256x128xf32>, %arg6: tensor<128xf32>) -> tensor<128x128xf32> attributes {llvm.emit_c_interface} {
    %0 = tensor.empty() : tensor<128x512xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x512xf32>) -> tensor<128x512xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<128x13xf32>, tensor<13x512xf32>) outs(%1 : tensor<128x512xf32>) -> tensor<128x512xf32>
    %3 = tensor.empty() : tensor<128x512xf32>
    %broadcasted = linalg.broadcast ins(%arg2 : tensor<512xf32>) outs(%3 : tensor<128x512xf32>) dimensions = [0]
    %4 = tensor.empty() : tensor<128x512xf32>
    %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2, %broadcasted : tensor<128x512xf32>, tensor<128x512xf32>) outs(%4 : tensor<128x512xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %24 = arith.addf %in, %in_4 : f32
      linalg.yield %24 : f32
    } -> tensor<128x512xf32>
    %6 = tensor.empty() : tensor<128x512xf32>
    %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<128x512xf32>) outs(%6 : tensor<128x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_4 = arith.constant 0.000000e+00 : f32
      %24 = arith.maximumf %in, %cst_4 : f32
      linalg.yield %24 : f32
    } -> tensor<128x512xf32>
    %8 = tensor.empty() : tensor<128x256xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %9 = linalg.fill ins(%cst_0 : f32) outs(%8 : tensor<128x256xf32>) -> tensor<128x256xf32>
    %10 = linalg.matmul ins(%7, %arg3 : tensor<128x512xf32>, tensor<512x256xf32>) outs(%9 : tensor<128x256xf32>) -> tensor<128x256xf32>
    %11 = tensor.empty() : tensor<128x256xf32>
    %broadcasted_1 = linalg.broadcast ins(%arg4 : tensor<256xf32>) outs(%11 : tensor<128x256xf32>) dimensions = [0]
    %12 = tensor.empty() : tensor<128x256xf32>
    %13 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%10, %broadcasted_1 : tensor<128x256xf32>, tensor<128x256xf32>) outs(%12 : tensor<128x256xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %24 = arith.addf %in, %in_4 : f32
      linalg.yield %24 : f32
    } -> tensor<128x256xf32>
    %14 = tensor.empty() : tensor<128x256xf32>
    %15 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%13 : tensor<128x256xf32>) outs(%14 : tensor<128x256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_4 = arith.constant 0.000000e+00 : f32
      %24 = arith.maximumf %in, %cst_4 : f32
      linalg.yield %24 : f32
    } -> tensor<128x256xf32>
    %16 = tensor.empty() : tensor<128x128xf32>
    %cst_2 = arith.constant 0.000000e+00 : f32
    %17 = linalg.fill ins(%cst_2 : f32) outs(%16 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %18 = linalg.matmul ins(%15, %arg5 : tensor<128x256xf32>, tensor<256x128xf32>) outs(%17 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %19 = tensor.empty() : tensor<128x128xf32>
    %broadcasted_3 = linalg.broadcast ins(%arg6 : tensor<128xf32>) outs(%19 : tensor<128x128xf32>) dimensions = [0]
    %20 = tensor.empty() : tensor<128x128xf32>
    %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %broadcasted_3 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%20 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %24 = arith.addf %in, %in_4 : f32
      linalg.yield %24 : f32
    } -> tensor<128x128xf32>
    %22 = tensor.empty() : tensor<128x128xf32>
    %23 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21 : tensor<128x128xf32>) outs(%22 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_4 = arith.constant 0.000000e+00 : f32
      %24 = arith.maximumf %in, %cst_4 : f32
      linalg.yield %24 : f32
    } -> tensor<128x128xf32>
    return %23 : tensor<128x128xf32>
  }
}
