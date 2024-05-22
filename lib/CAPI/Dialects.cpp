#include "TPP-c/Dialects.h"
#include "TPP/Dialect/Check/CheckDialect.h"
#include "TPP/Dialect/Perf/PerfDialect.h"
#include "TPP/Dialect/Xsmm/XsmmDialect.h"
#include "mlir/CAPI/Registration.h"


MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Check, check,
                                      mlir::check::CheckDialect)

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Perf, perf,
                                      mlir::perf::PerfDialect)

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Xsmm, xsmm,
                                      mlir::xsmm::XsmmDialect)