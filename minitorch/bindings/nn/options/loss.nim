{.push header: "nn/options/loss.h".}


# Constructors and methods
proc TORCH_OPTIONS_CTOR_VARIANT_ARG3*(this: var L1LossOptions): int  {.importcpp: "TORCH_OPTIONS_CTOR_VARIANT_ARG3".}

proc TORCH_OPTIONS_CTOR_VARIANT_ARG4*(this: var KLDivLossOptions): int  {.importcpp: "TORCH_OPTIONS_CTOR_VARIANT_ARG4".}

proc TORCH_ARG*(this: var KLDivLossOptions): int  {.importcpp: "TORCH_ARG".}
  ## Specifies whether `target` is accepted in the log space. Default:
  ## False

proc TORCH_OPTIONS_CTOR_VARIANT_ARG3*(this: var MSELossOptions): int  {.importcpp: "TORCH_OPTIONS_CTOR_VARIANT_ARG3".}

proc TORCH_ARG*(this: var BCELossOptions): int  {.importcpp: "TORCH_ARG".}
  ## A manual rescaling weight given to the loss of each batch element.

proc TORCH_ARG*(this: var BCELossOptions): int  {.importcpp: "TORCH_ARG".}
  ## Specifies the reduction to apply to the output. ``'none'`` |
  ## ``'mean'`` | ``'sum'``. Default: ``'mean'``

proc TORCH_ARG*(this: var HingeEmbeddingLossOptions): int  {.importcpp: "TORCH_ARG".}
  ## Specifies the threshold for which the distance of a negative sample
  ## must reach in order to incur zero loss. Default: 1

proc TORCH_ARG*(this: var HingeEmbeddingLossOptions): int  {.importcpp: "TORCH_ARG".}
  ## Specifies the reduction to apply to the output. Default: Mean

proc TORCH_ARG*(this: var MultiMarginLossOptions): int  {.importcpp: "TORCH_ARG".}
  ## Has a default value of :math:`1`. :math:`1` and :math:`2` are the only
  ## supported values.

proc TORCH_ARG*(this: var MultiMarginLossOptions): int  {.importcpp: "TORCH_ARG".}
  ## Has a default value of :math:`1`.

proc TORCH_ARG*(this: var MultiMarginLossOptions): int  {.importcpp: "TORCH_ARG".}
  ## A manual rescaling weight given to each class. If given, it has to be
  ## a Tensor of size `C`. Otherwise, it is treated as if having all ones.

proc TORCH_ARG*(this: var MultiMarginLossOptions): int  {.importcpp: "TORCH_ARG".}
  ## Specifies the reduction to apply to the output: ``'none'`` |
  ## ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
  ## ``'mean'``: the sum of the output will be divided by the number of
  ## elements in the output, ``'sum'``: the output will be summed. Default:
  ## ``'mean'``

proc TORCH_ARG*(this: var CosineEmbeddingLossOptions): int  {.importcpp: "TORCH_ARG".}
  ## Specifies the threshold for which the distance of a negative sample
  ## must reach in order to incur zero loss. Should be a number from -1 to
  ## 1, 0 to 0.5 is suggested. Default: 0.0

proc TORCH_ARG*(this: var CosineEmbeddingLossOptions): int  {.importcpp: "TORCH_ARG".}
  ## Specifies the reduction to apply to the output. Default: Mean

proc TORCH_OPTIONS_CTOR_VARIANT_ARG3*(this: var MultiLabelMarginLossOptions): int  {.importcpp: "TORCH_OPTIONS_CTOR_VARIANT_ARG3".}

proc TORCH_OPTIONS_CTOR_VARIANT_ARG3*(this: var SoftMarginLossOptions): int  {.importcpp: "TORCH_OPTIONS_CTOR_VARIANT_ARG3".}

proc TORCH_ARG*(this: var MultiLabelSoftMarginLossOptions): int  {.importcpp: "TORCH_ARG".}
  ## A manual rescaling weight given to each class. If given, it has to be
  ## a Tensor of size `C`. Otherwise, it is treated as if having all ones.

proc TORCH_ARG*(this: var MultiLabelSoftMarginLossOptions): int  {.importcpp: "TORCH_ARG".}
  ## Specifies the reduction to apply to the output: 'none' | 'mean' |
  ## 'sum'. 'none': no reduction will be applied, 'mean': the sum of the
  ## output will be divided by the number of elements in the output, 'sum':
  ## the output will be summed. Default: 'mean'

proc TORCH_ARG*(this: var TripletMarginLossOptions): int  {.importcpp: "TORCH_ARG".}
  ## Specifies the threshold for which the distance of a negative sample
  ## must reach in order to incur zero loss. Default: 1

proc TORCH_ARG*(this: var TripletMarginLossOptions): int  {.importcpp: "TORCH_ARG".}
  ## Specifies the norm degree for pairwise distance. Default: 2

proc TORCH_ARG*(this: var TripletMarginLossOptions): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var TripletMarginLossOptions): int  {.importcpp: "TORCH_ARG".}
  ## The distance swap is described in detail in the paper Learning shallow
  ## convolutional feature descriptors with triplet losses by V. Balntas,
  ## E. Riba et al. Default: False

proc TORCH_ARG*(this: var TripletMarginLossOptions): int  {.importcpp: "TORCH_ARG".}
  ## Specifies the reduction to apply to the output. Default: Mean

proc TORCH_ARG*(this: var CTCLossOptions): int  {.importcpp: "TORCH_ARG".}
  ## blank label. Default `0`.

proc TORCH_ARG*(this: var CTCLossOptions): int  {.importcpp: "TORCH_ARG".}
  ## Specifies the reduction to apply to the output. Default: Mean

proc TORCH_ARG*(this: var CTCLossOptions): int  {.importcpp: "TORCH_ARG".}
  ## Whether to zero infinite losses and the associated gradients. Default:
  ## `false`. Infinite losses mainly occur when the inputs are too short to
  ## be aligned to the targets.

proc TORCH_OPTIONS_CTOR_VARIANT_ARG3*(this: var SmoothL1LossOptions): int  {.importcpp: "TORCH_OPTIONS_CTOR_VARIANT_ARG3".}

proc TORCH_ARG*(this: var PoissonNLLLossOptions): int  {.importcpp: "TORCH_ARG".}
  ## if true the loss is computed as `exp(input) - target * input`, if
  ## false the loss is `input - target * log(input + eps)`.

proc TORCH_ARG*(this: var PoissonNLLLossOptions): int  {.importcpp: "TORCH_ARG".}
  ## whether to compute full loss, i.e. to add the Stirling approximation
  ## term target * log(target) - target + 0.5 * log(2 * pi * target).

proc TORCH_ARG*(this: var PoissonNLLLossOptions): int  {.importcpp: "TORCH_ARG".}
  ## Small value to avoid evaluation of `log(0)` when `log_input = false`.
  ## Default: 1e-8

proc TORCH_ARG*(this: var PoissonNLLLossOptions): int  {.importcpp: "TORCH_ARG".}
  ## Specifies the reduction to apply to the output. Default: Mean

proc TORCH_ARG*(this: var MarginRankingLossOptions): int  {.importcpp: "TORCH_ARG".}
  ## Has a default value of `0`.

proc TORCH_ARG*(this: var MarginRankingLossOptions): int  {.importcpp: "TORCH_ARG".}
  ## Specifies the reduction to apply to the output. Default: Mean

proc TORCH_ARG*(this: var NLLLossOptions): int  {.importcpp: "TORCH_ARG".}
  ## A manual rescaling weight given to each class. If given, it has to be
  ## a Tensor of size `C`. Otherwise, it is treated as if having all ones.

proc TORCH_ARG*(this: var NLLLossOptions): int  {.importcpp: "TORCH_ARG".}
  ## Specifies a target value that is ignored and does not contribute to
  ## the input gradient.

proc TORCH_ARG*(this: var NLLLossOptions): int  {.importcpp: "TORCH_ARG".}
  ## Specifies the reduction to apply to the output. Default: Mean

proc TORCH_ARG*(this: var CrossEntropyLossOptions): int  {.importcpp: "TORCH_ARG".}
  ## A manual rescaling weight given to each class. If given, has to be a
  ## Tensor of size C

proc TORCH_ARG*(this: var CrossEntropyLossOptions): int  {.importcpp: "TORCH_ARG".}
  ## Specifies a target value that is ignored and does not contribute to
  ## the input gradient.

proc TORCH_ARG*(this: var CrossEntropyLossOptions): int  {.importcpp: "TORCH_ARG".}
  ## Specifies the reduction to apply to the output. Default: Mean

proc TORCH_ARG*(this: var BCEWithLogitsLossOptions): int  {.importcpp: "TORCH_ARG".}
  ## A manual rescaling weight given to the loss of each batch element. If
  ## given, has to be a Tensor of size `nbatch`.

proc TORCH_ARG*(this: var BCEWithLogitsLossOptions): int  {.importcpp: "TORCH_ARG".}
  ## Specifies the reduction to apply to the output. Default: Mean

proc TORCH_ARG*(this: var BCEWithLogitsLossOptions): int  {.importcpp: "TORCH_ARG".}
  ## A weight of positive examples. Must be a vector with length equal to
  ## the number of classes.

{.pop.} # header: "nn/options/loss.h
