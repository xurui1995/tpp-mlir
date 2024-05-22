include(AddMLIRPython)

# Specifies that all MLIR packages are co-located under the `tpp`
# top level package (the API has been embedded in a relocatable way).
# TODO: Add an upstream cmake param for this vs having a global here.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=tpp.")


################################################################################
# Sources
################################################################################

declare_mlir_python_sources(TppPythonSources)


################################################################################
# Dialect bindings
################################################################################

declare_mlir_dialect_python_bindings(
    ADD_TO_PARENT TppPythonSources
    ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tpp"
    TD_FILE dialects/CheckOps.td
    SOURCES 
      dialects/check.py
    DIALECT_NAME check
    )

declare_mlir_dialect_python_bindings(
      ADD_TO_PARENT TppPythonSources
      ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tpp"
      TD_FILE dialects/PerfOps.td
      SOURCES 
        dialects/perf.py
      DIALECT_NAME perf
      )

declare_mlir_dialect_python_bindings(
        ADD_TO_PARENT TppPythonSources
        ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tpp"
        TD_FILE dialects/XsmmOps.td
        SOURCES 
          dialects/xsmm.py
        DIALECT_NAME xsmm
        )

declare_mlir_python_extension(TppPythonSources.Extension
  MODULE_NAME _tpp
  ADD_TO_PARENT TppPythonSources
  SOURCES
    MainModule.cpp
  EMBED_CAPI_LINK_LIBS
    TPPCAPI
)

################################################################################
# Common CAPI
################################################################################

add_mlir_python_common_capi_library(TppPythonCAPI
  INSTALL_COMPONENT TppPythonModules
  INSTALL_DESTINATION python_packages/tpp_core/tpp/_mlir_libs
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/python_packages/tpp_core/tpp/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    TppPythonSources
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.ExecutionEngine
    MLIRPythonSources.Core
)

################################################################################
# Instantiation of all Python modules
################################################################################

add_mlir_python_modules(TppPythonModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/python_packages/tpp_core/tpp"
  INSTALL_PREFIX "python_packages/tpp_core/tpp"
  DECLARED_SOURCES
    TppPythonSources
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.ExecutionEngine
    MLIRPythonSources
  COMMON_CAPI_LINK_LIBS
    TppPythonCAPI
  )