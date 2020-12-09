{.push header: "nn/options/transformerlayer.h".}


# Constructors and methods
proc constructor_TransformerEncoderLayerOptions*(d_model: cint, nhead: cint): TransformerEncoderLayerOptions {.constructor,importcpp: "TransformerEncoderLayerOptions(@)".}

proc constructor_TransformerDecoderLayerOptions*(d_model: cint, nhead: cint): TransformerDecoderLayerOptions {.constructor,importcpp: "TransformerDecoderLayerOptions(@)".}

proc TORCH_ARG*(this: var TransformerEncoderLayerOptions): int  {.importcpp: "TORCH_ARG".}
  ## the number of expected features in the input

proc TORCH_ARG*(this: var TransformerEncoderLayerOptions): int  {.importcpp: "TORCH_ARG".}
  ## the number of heads in the multiheadattention models

proc TORCH_ARG*(this: var TransformerEncoderLayerOptions): int  {.importcpp: "TORCH_ARG".}
  ## the dimension of the feedforward network model, default is 2048

proc TORCH_ARG*(this: var TransformerEncoderLayerOptions): int  {.importcpp: "TORCH_ARG".}
  ## the dropout value, default is 0.1

proc TORCH_ARG*(this: var TransformerEncoderLayerOptions): int  {.importcpp: "TORCH_ARG".}
  ## the activation function of intermediate layer, either ``torch::kReLU``
  ## or ``torch::GELU``, default is ``torch::kReLU``

proc TORCH_ARG*(this: var TransformerDecoderLayerOptions): int  {.importcpp: "TORCH_ARG".}
  ## number of expected features in the input

proc TORCH_ARG*(this: var TransformerDecoderLayerOptions): int  {.importcpp: "TORCH_ARG".}
  ## number of heads in the multiheadattention models

proc TORCH_ARG*(this: var TransformerDecoderLayerOptions): int  {.importcpp: "TORCH_ARG".}
  ## dimension of the feedforward network model. Default: 2048

proc TORCH_ARG*(this: var TransformerDecoderLayerOptions): int  {.importcpp: "TORCH_ARG".}
  ## dropout value. Default: 1

proc TORCH_ARG*(this: var TransformerDecoderLayerOptions): int  {.importcpp: "TORCH_ARG".}
  ## activation function of intermediate layer, can be either
  ## ``torch::kGELU`` or ``torch::kReLU``. Default: ``torch::kReLU``

{.pop.} # header: "nn/options/transformerlayer.h
