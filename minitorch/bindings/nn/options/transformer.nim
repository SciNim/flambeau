{.push header: "nn/options/transformer.h".}


# Constructors and methods
proc constructor_TransformerOptions*(): TransformerOptions {.constructor,importcpp: "TransformerOptions".}

proc constructor_TransformerOptions*(d_model: cint, nhead: cint): TransformerOptions {.constructor,importcpp: "TransformerOptions(@)".}

proc constructor_TransformerOptions*(d_model: cint, nhead: cint, num_encoder_layers: cint, num_decoder_layers: cint): TransformerOptions {.constructor,importcpp: "TransformerOptions(@)".}

proc TORCH_ARG*(this: var TransformerOptions): int  {.importcpp: "TORCH_ARG".}
  ## the number of expected features in the encoder/decoder inputs
  ## (default=512)

proc TORCH_ARG*(this: var TransformerOptions): int  {.importcpp: "TORCH_ARG".}
  ## the number of heads in the multiheadattention models (default=8)

proc TORCH_ARG*(this: var TransformerOptions): int  {.importcpp: "TORCH_ARG".}
  ## the number of sub-encoder-layers in the encoder (default=6)

proc TORCH_ARG*(this: var TransformerOptions): int  {.importcpp: "TORCH_ARG".}
  ## the number of sub-decoder-layers in the decoder (default=6)

proc TORCH_ARG*(this: var TransformerOptions): int  {.importcpp: "TORCH_ARG".}
  ## the dimension of the feedforward network model (default=2048)

proc TORCH_ARG*(this: var TransformerOptions): int  {.importcpp: "TORCH_ARG".}
  ## the dropout value (default=0.1)

proc TORCH_ARG*(this: var TransformerOptions): int  {.importcpp: "TORCH_ARG".}
  ## the activation function of encoder/decoder intermediate layer
  ## (default=``torch::kReLU``)

proc TORCH_ARG*(this: var TransformerOptions): int  {.importcpp: "TORCH_ARG".}
  ## custom encoder (default=None)

proc TORCH_ARG*(this: var TransformerOptions): int  {.importcpp: "TORCH_ARG".}
  ## custom decoder (default=None)

{.pop.} # header: "nn/options/transformer.h
