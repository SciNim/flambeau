{.push header: "nn/options/vision.h".}


# Constructors and methods
proc TORCH_ARG*(this: var GridSampleFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## interpolation mode to calculate output values. Default: Bilinear

proc TORCH_ARG*(this: var GridSampleFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## padding mode for outside grid values. Default: Zeros

{.pop.} # header: "nn/options/vision.h
