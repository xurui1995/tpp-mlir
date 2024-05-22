add_subdirectory(Dialect)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name TppCompiler)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix TppCompiler)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix TppCompiler)
add_public_tablegen_target(TPPCompilerPassIncGen)
add_mlir_doc(Passes TppCompilerPasses ./ -gen-pass-doc)
