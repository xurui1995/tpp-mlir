#include "TPP/Passes.h"
#include "mlir-c/Pass.h"
#include "mlir/CAPI/Pass.h"

#include "TPP/Passes.capi.h.inc"
using namespace mlir;
using namespace mlir::tpp;


#ifdef __cplusplus
extern "C" {
#endif

#include "TPP/Passes.capi.cpp.inc"

#ifdef __cplusplus
}
#endif