//===- TensorInit.cpp --------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Transforms/Utils/TensorInit.h"
#include "TPP/Transforms/Utils/TensorInitFloat.h"
#include "TPP/Transforms/Utils/TensorInitInt.h"

#include <functional>
#include <unordered_map>
#include <iostream>

using namespace mlir;

namespace {

struct InitKey {
  InitKey() = default;
  explicit InitKey(TensorInitType type, mlir::Type elmType, int seed)
      : type(type) {
    // Seed only matters for randomized types.
    switch (type) {
    case TensorInitType::Random:
    case TensorInitType::Normal:
      this->seed = seed;
      break;
    default:
      this->seed = 0;
      break;
    }

    floatType = TensorInitFloat::getTensorInitDataType(elmType);
    intType = TensorInitInt::getTensorInitDataType(elmType);
  }

  bool operator==(const InitKey &ik) const {
    return type == ik.type && floatType == ik.floatType &&
           intType == ik.intType && seed == ik.seed;
  }

  TensorInitType type;
  TensorInitFloat::DataType floatType;
  TensorInitInt::DataType intType;
  int seed;
};

struct InitKeyHash_fn {
  std::size_t operator()(const InitKey &ik) const {
    std::size_t h1 = std::hash<TensorInitType>{}(ik.type);
    std::size_t h2 = std::hash<TensorInitFloat::DataType>{}(ik.floatType);
    std::size_t h3 = std::hash<TensorInitInt::DataType>{}(ik.intType);
    std::size_t h4 = std::hash<int>{}(ik.seed);
    return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3);
  }
};

std::unordered_map<InitKey, TensorInitPtr, InitKeyHash_fn> tensorInitializers;
} // namespace

TensorInitType parseTensorInitType(StringRef name) {
  auto type = StringSwitch<TensorInitType>(name)
                  .Case("", TensorInitType::Auto)
                  .Case("const", TensorInitType::Constant)
                  .Case("simple", TensorInitType::Simple)
                  .Case("cont", TensorInitType::Continuous)
                  .Case("random", TensorInitType::Random)
                  .Case("normal", TensorInitType::Normal)
                  .Default(TensorInitType::Invalid);
  return type;
}

TensorInitPtr getTensorInit(TensorInitType type, mlir::Type elmType, int seed) {
  // Defaults for seed or not
  if (type == TensorInitType::Auto) {
    if (seed)
      type = TensorInitType::Normal;
    else
      type = TensorInitType::Constant;
  }

  InitKey key(type, elmType, seed);
  if (tensorInitializers.find(key) != tensorInitializers.end())
    return tensorInitializers[key];

  TensorInitPtr initPtr = nullptr;

  if (TensorInitFloat::isTypeSupported(elmType)) {
    auto dataType = TensorInitFloat::getTensorInitDataType(elmType);
    switch (type) {
    case TensorInitType::Constant:
      std::cout << "1" << std::endl;
      initPtr = std::make_shared<ConstantTensorInitFloat>(dataType);
      break;
    case TensorInitType::Simple:
     std::cout << "2" << std::endl;
      initPtr = std::make_shared<SimpleTensorInitFloat>(dataType);
      break;
    case TensorInitType::Continuous:
     std::cout << "3" << std::endl;
      initPtr = std::make_shared<ContinuousTensorInitFloat>(dataType);
      break;
    case TensorInitType::Random:
     std::cout << "4" << std::endl;
      assert(seed && "Can't call random initializers without seed");
      initPtr = std::make_shared<RandomTensorInitFloat>(dataType, seed);
      break;
    case TensorInitType::Normal:
     std::cout << "5" << std::endl;
      assert(seed && "Can't call random initializers without seed");
      initPtr = std::make_shared<NormalTensorInitFloat>(dataType, seed);
      break;
    default:
      assert(false && "Invalid tensor initializer type");
    }
  }

  if (TensorInitInt::isTypeSupported(elmType)) {
    auto dataType = TensorInitInt::getTensorInitDataType(elmType);
    switch (type) {
       
    case TensorInitType::Constant:
    std::cout << "6" << std::endl;
      initPtr = std::make_shared<ConstantTensorInitInt>(dataType);
      break;
    case TensorInitType::Simple:
      std::cout << "7" << std::endl;
      initPtr = std::make_shared<SimpleTensorInitInt>(dataType);
      break;
    case TensorInitType::Continuous:
      std::cout << "8" << std::endl;
      initPtr = std::make_shared<ContinuousTensorInitInt>(dataType);
      break;
    case TensorInitType::Random:
      std::cout << "9" << std::endl;
      assert(seed && "Can't call random initializers without seed");
      initPtr = std::make_shared<RandomTensorInitInt>(dataType, seed);
      break;
    case TensorInitType::Normal:
      std::cout << "10" << std::endl;
      assert(seed && "Can't call random initializers without seed");
      initPtr = std::make_shared<NormalTensorInitInt>(dataType, seed);
      break;
    default:
      assert(false && "Invalid tensor initializer type");
    }
  }

  assert(initPtr && "Unsupported tensor element type");
  tensorInitializers[key] = initPtr;

  return initPtr;
}

TensorInitPtr getTensorInit(StringRef type, mlir::Type elmType, int seed) {
  auto initType = parseTensorInitType(type);
  return getTensorInit(initType, elmType, seed);
}
