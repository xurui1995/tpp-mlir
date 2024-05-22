#ifndef TPP_MLIR_C_DIALECTS_H
#define TPP_MLIR_C_DIALECTS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Check, check);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Perf, perf);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Xsmm, xsmm);


#ifdef __cplusplus
}
#endif
#endif // GC_MLIR_C_DIALECTS_H