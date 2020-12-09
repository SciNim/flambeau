{.push header: "nn/options/instancenorm.h".}


# Constructors and methods
proc constructor_InstanceNormOptions*(num_features: cint): InstanceNormOptions {.constructor,importcpp: "InstanceNormOptions(@)".}

proc TORCH_ARG*(this: var InstanceNormOptions): int  {.importcpp: "TORCH_ARG".}
  ## The number of features of the input tensor.

proc TORCH_ARG*(this: var InstanceNormOptions): int  {.importcpp: "TORCH_ARG".}
  ## The epsilon value added for numerical stability.

proc TORCH_ARG*(this: var InstanceNormOptions): int  {.importcpp: "TORCH_ARG".}
  ## A momentum multiplier for the mean and variance.

proc TORCH_ARG*(this: var InstanceNormOptions): int  {.importcpp: "TORCH_ARG".}
  ## Whether to learn a scale and bias that are applied in an affine
  ## transformation on the input.

proc TORCH_ARG*(this: var InstanceNormOptions): int  {.importcpp: "TORCH_ARG".}
  ## Whether to store and update batch statistics (mean and variance) in
  ## the module.

proc TORCH_ARG*(this: var InstanceNormFuncOptions): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var InstanceNormFuncOptions): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var InstanceNormFuncOptions): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var InstanceNormFuncOptions): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var InstanceNormFuncOptions): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var InstanceNormFuncOptions): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var InstanceNormFuncOptions): int  {.importcpp: "TORCH_ARG".}

{.pop.} # header: "nn/options/instancenorm.h
