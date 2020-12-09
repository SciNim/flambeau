{.push header: "nn/options/normalization.h".}


# Constructors and methods
proc constructor_LayerNormOptions*(normalized_shape: cint): LayerNormOptions {.constructor,importcpp: "LayerNormOptions(@)".}

proc constructor_LayerNormFuncOptions*(normalized_shape: cint): LayerNormFuncOptions {.constructor,importcpp: "LayerNormFuncOptions(@)".}

proc constructor_LocalResponseNormOptions*(size: cint): LocalResponseNormOptions {.constructor,importcpp: "LocalResponseNormOptions(@)".}

proc constructor_CrossMapLRN2dOptions*(size: cint): CrossMapLRN2dOptions {.constructor,importcpp: "CrossMapLRN2dOptions(@)".}

proc constructor_GroupNormOptions*(num_groups: cint, num_channels: cint): GroupNormOptions {.constructor,importcpp: "GroupNormOptions(@)".}

proc constructor_GroupNormFuncOptions*(num_groups: cint): GroupNormFuncOptions {.constructor,importcpp: "GroupNormFuncOptions(@)".}

proc TORCH_ARG*(this: var LayerNormOptions): int  {.importcpp: "TORCH_ARG".}
  ## input shape from an expected input.

proc TORCH_ARG*(this: var LayerNormOptions): int  {.importcpp: "TORCH_ARG".}
  ## a value added to the denominator for numerical stability. ``Default:
  ## 1e-5``.

proc TORCH_ARG*(this: var LayerNormOptions): int  {.importcpp: "TORCH_ARG".}
  ## a boolean value that when set to ``true``, this module has learnable
  ## per-element affine parameters initialized to ones (for weights) and
  ## zeros (for biases). ``Default: true``.

proc TORCH_ARG*(this: var LayerNormFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## input shape from an expected input.

proc TORCH_ARG*(this: var LayerNormFuncOptions): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var LayerNormFuncOptions): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var LayerNormFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## a value added to the denominator for numerical stability. ``Default:
  ## 1e-5``.

proc TORCH_ARG*(this: var LocalResponseNormOptions): int  {.importcpp: "TORCH_ARG".}
  ## amount of neighbouring channels used for normalization

proc TORCH_ARG*(this: var LocalResponseNormOptions): int  {.importcpp: "TORCH_ARG".}
  ## multiplicative factor. Default: 1e-4

proc TORCH_ARG*(this: var LocalResponseNormOptions): int  {.importcpp: "TORCH_ARG".}
  ## exponent. Default: 0.75

proc TORCH_ARG*(this: var LocalResponseNormOptions): int  {.importcpp: "TORCH_ARG".}
  ## additive factor. Default: 1

proc TORCH_ARG*(this: var CrossMapLRN2dOptions): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var CrossMapLRN2dOptions): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var CrossMapLRN2dOptions): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var CrossMapLRN2dOptions): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var NormalizeFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## The exponent value in the norm formulation. Default: 2.0

proc TORCH_ARG*(this: var NormalizeFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## The dimension to reduce. Default: 1

proc TORCH_ARG*(this: var NormalizeFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## Small value to avoid division by zero. Default: 1e-12

proc TORCH_ARG*(this: var GroupNormOptions): int  {.importcpp: "TORCH_ARG".}
  ## number of groups to separate the channels into

proc TORCH_ARG*(this: var GroupNormOptions): int  {.importcpp: "TORCH_ARG".}
  ## number of channels expected in input

proc TORCH_ARG*(this: var GroupNormOptions): int  {.importcpp: "TORCH_ARG".}
  ## a value added to the denominator for numerical stability. Default:
  ## 1e-5

proc TORCH_ARG*(this: var GroupNormOptions): int  {.importcpp: "TORCH_ARG".}
  ## a boolean value that when set to ``true``, this module has learnable
  ## per-channel affine parameters initialized to ones (for weights) and
  ## zeros (for biases). Default: ``true``.

proc TORCH_ARG*(this: var GroupNormFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## number of groups to separate the channels into

proc TORCH_ARG*(this: var GroupNormFuncOptions): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var GroupNormFuncOptions): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var GroupNormFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## a value added to the denominator for numerical stability. Default:
  ## 1e-5

{.pop.} # header: "nn/options/normalization.h
