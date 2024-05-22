message (STATUS "===========: ${XSMM_INCLUDE_DIRS}")

add_mlir_public_c_api_library(TPPCAPI
  Dialects.cpp
  Passes.cpp
  
  LINK_LIBS PUBLIC
  TPPCheckDialect
  TPPPerfDialect
  TPPXsmmDialect
  TPPLinalgToFunc
  TPPLinalgToXSMM
  TPPPerfToFunc
  TPPPerfToLoop
  TPPXsmmToFunc
  TPPTransformsUtils
  TPPTransforms
  TPPGPU
  TPPPipeline
  TPPRunner

)