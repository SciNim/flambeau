{.push header: "nn/options/batchnorm.h".}


# Constructors and methods
proc constructor_BatchNormOptions*(num_features: cint): BatchNormOptions {.constructor,importcpp: "BatchNormOptions(@)".}

proc TORCH_ARG*(this: var BatchNormOptions): int  {.importcpp: "TORCH_ARG".}
  ## The number of features of the input tensor. Changing this parameter
  ## after construction __has no effect__.

proc TORCH_ARG*(this: var BatchNormOptions): int  {.importcpp: "TORCH_ARG".}
  ## The epsilon value added for numerical stability. Changing this
  ## parameter after construction __is effective__.

proc TORCH_ARG*(this: var BatchNormOptions): int  {.importcpp: "TORCH_ARG".}
  ## Whether to learn a scale and bias that are applied in an affine
  ## transformation on the input. Changing this parameter after
  ## construction __has no effect__.

proc TORCH_ARG*(this: var BatchNormOptions): int  {.importcpp: "TORCH_ARG".}
  ## Whether to store and update batch statistics (mean and variance) in
  ## the module. Changing this parameter after construction __has no
  ## effect__.

proc TORCH_ARG*(this: var BatchNormFuncOptions): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var BatchNormFuncOptions): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var BatchNormFuncOptions): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var BatchNormFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## The epsilon value added for numerical stability. Changing this
  ## parameter after construction __is effective__.

{.pop.} # header: "nn/options/batchnorm.h
