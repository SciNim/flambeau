{.push header: "nn/options/linear.h".}


# Constructors and methods
proc constructor_LinearOptions*(in_features: int64_t, out_features: int64_t): LinearOptions {.constructor,importcpp: "LinearOptions(@)".}

proc constructor_UnflattenOptions*(dim: int64_t, sizes: cint): UnflattenOptions {.constructor,importcpp: "UnflattenOptions(@)".}

proc constructor_UnflattenOptions*(dimname: char *, namedshape: torch::nn::UnflattenOptions::namedshape_t): UnflattenOptions {.constructor,importcpp: "UnflattenOptions(@)".}

proc constructor_UnflattenOptions*(dimname: std::string, namedshape: torch::nn::UnflattenOptions::namedshape_t): UnflattenOptions {.constructor,importcpp: "UnflattenOptions(@)".}

proc constructor_BilinearOptions*(in1_features: int64_t, in2_features: int64_t, out_features: int64_t): BilinearOptions {.constructor,importcpp: "BilinearOptions(@)".}

proc TORCH_ARG*(this: var LinearOptions): int  {.importcpp: "TORCH_ARG".}
  ## size of each input sample

proc TORCH_ARG*(this: var LinearOptions): int  {.importcpp: "TORCH_ARG".}
  ## size of each output sample

proc TORCH_ARG*(this: var LinearOptions): int  {.importcpp: "TORCH_ARG".}
  ## If set to false, the layer will not learn an additive bias. Default:
  ## true

proc TORCH_ARG*(this: var FlattenOptions): int  {.importcpp: "TORCH_ARG".}
  ## first dim to flatten

proc TORCH_ARG*(this: var FlattenOptions): int  {.importcpp: "TORCH_ARG".}
  ## last dim to flatten

proc TORCH_ARG*(this: var UnflattenOptions): int  {.importcpp: "TORCH_ARG".}
  ## dim to unflatten

proc TORCH_ARG*(this: var UnflattenOptions): int  {.importcpp: "TORCH_ARG".}
  ## name of dim to unflatten, for use with named tensors

proc TORCH_ARG*(this: var UnflattenOptions): int  {.importcpp: "TORCH_ARG".}
  ## new shape of unflattened dim

proc TORCH_ARG*(this: var UnflattenOptions): int  {.importcpp: "TORCH_ARG".}
  ## new shape of unflattened dim with names, for use with named tensors

proc TORCH_ARG*(this: var BilinearOptions): int  {.importcpp: "TORCH_ARG".}
  ## The number of features in input 1 (columns of the input1 matrix).

proc TORCH_ARG*(this: var BilinearOptions): int  {.importcpp: "TORCH_ARG".}
  ## The number of features in input 2 (columns of the input2 matrix).

proc TORCH_ARG*(this: var BilinearOptions): int  {.importcpp: "TORCH_ARG".}
  ## The number of output features to produce (columns of the output
  ## matrix).

proc TORCH_ARG*(this: var BilinearOptions): int  {.importcpp: "TORCH_ARG".}
  ## Whether to learn and add a bias after the bilinear transformation.

{.pop.} # header: "nn/options/linear.h
