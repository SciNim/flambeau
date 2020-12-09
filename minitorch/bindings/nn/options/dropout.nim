{.push header: "nn/options/dropout.h".}


# Constructors and methods
proc constructor_DropoutOptions*(p: cdouble): DropoutOptions {.constructor,importcpp: "DropoutOptions(@)".}

proc TORCH_ARG*(this: var DropoutOptions): int  {.importcpp: "TORCH_ARG".}
  ## The probability of an element to be zeroed. Default: 0.5

proc TORCH_ARG*(this: var DropoutOptions): int  {.importcpp: "TORCH_ARG".}
  ## can optionally do the operation in-place. Default: False

proc TORCH_ARG*(this: var DropoutFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## The probability of an element to be zeroed. Default: 0.5

proc TORCH_ARG*(this: var DropoutFuncOptions): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var DropoutFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## can optionally do the operation in-place. Default: False

proc TORCH_ARG*(this: var AlphaDropoutFuncOptions): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var AlphaDropoutFuncOptions): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var AlphaDropoutFuncOptions): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var FeatureAlphaDropoutFuncOptions): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var FeatureAlphaDropoutFuncOptions): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var FeatureAlphaDropoutFuncOptions): int  {.importcpp: "TORCH_ARG".}

{.pop.} # header: "nn/options/dropout.h
