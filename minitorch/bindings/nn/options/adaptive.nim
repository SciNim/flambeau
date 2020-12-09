{.push header: "nn/options/adaptive.h".}


# Constructors and methods
proc constructor_AdaptiveLogSoftmaxWithLossOptions*(in_features: cint, n_classes: cint, cutoffs: cint): AdaptiveLogSoftmaxWithLossOptions {.constructor,importcpp: "AdaptiveLogSoftmaxWithLossOptions(@)".}

proc TORCH_ARG*(this: var AdaptiveLogSoftmaxWithLossOptions): int  {.importcpp: "TORCH_ARG".}
  ## Number of features in the input tensor

proc TORCH_ARG*(this: var AdaptiveLogSoftmaxWithLossOptions): int  {.importcpp: "TORCH_ARG".}
  ## Number of classes in the dataset

proc TORCH_ARG*(this: var AdaptiveLogSoftmaxWithLossOptions): int  {.importcpp: "TORCH_ARG".}
  ## value used as an exponent to compute sizes of the clusters. Default:
  ## 4.0

proc TORCH_ARG*(this: var AdaptiveLogSoftmaxWithLossOptions): int  {.importcpp: "TORCH_ARG".}
  ## If ``true``, adds a bias term to the 'head' of the adaptive softmax.
  ## Default: false

{.pop.} # header: "nn/options/adaptive.h
