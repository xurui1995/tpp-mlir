

#include "TPP-c/Dialects.h"
#include "TPP-c/Passes.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include<iostream>

PYBIND11_MODULE(_tpp, m) {
  m.doc() = "Graph-compiler MLIR Python binding";

   mlirRegisterTppCompilerPasses();

  auto check = m.def_submodule("check");
  check.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle dialect = mlirGetDialectHandle__check__();
        mlirDialectHandleRegisterDialect(dialect, context);
        if (load) {
          mlirDialectHandleLoadDialect(dialect, context);
        }
      },
      py::arg("context") = py::none(), py::arg("load") = true);



  auto perf = m.def_submodule("perf");
  perf.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle dialect = mlirGetDialectHandle__perf__();
        mlirDialectHandleRegisterDialect(dialect, context);
        if (load) {
          mlirDialectHandleLoadDialect(dialect, context);
        }
      },
      py::arg("context") = py::none(), py::arg("load") = true);


    auto xsmm = m.def_submodule("xsmm");
  xsmm.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle dialect = mlirGetDialectHandle__xsmm__();
        mlirDialectHandleRegisterDialect(dialect, context);
        if (load) {
          mlirDialectHandleLoadDialect(dialect, context);
        }
      },
      py::arg("context") = py::none(), py::arg("load") = true);


  //===----------------------------------------------------------------------===//
  // OneDNNGraph
  //===----------------------------------------------------------------------===//
}