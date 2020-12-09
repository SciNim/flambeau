{.push header: "nn/options/transformercoder.h".}


# Constructors and methods
proc constructor_TransformerEncoderOptions*(encoder_layer: cint, num_layers: cint): TransformerEncoderOptions {.constructor,importcpp: "TransformerEncoderOptions(@)".}

proc constructor_TransformerEncoderOptions*(encoder_layer_options: cint, num_layers: cint): TransformerEncoderOptions {.constructor,importcpp: "TransformerEncoderOptions(@)".}

proc constructor_TransformerDecoderOptions*(decoder_layer: cint, num_layers: cint): TransformerDecoderOptions {.constructor,importcpp: "TransformerDecoderOptions(@)".}

proc constructor_TransformerDecoderOptions*(decoder_layer_options: cint, num_layers: cint): TransformerDecoderOptions {.constructor,importcpp: "TransformerDecoderOptions(@)".}

proc TORCH_ARG*(this: var TransformerEncoderOptions): int  {.importcpp: "TORCH_ARG".}
  ## transformer Encoder Layer

proc TORCH_ARG*(this: var TransformerEncoderOptions): int  {.importcpp: "TORCH_ARG".}
  ## number of encoder layers

proc TORCH_ARG*(this: var TransformerEncoderOptions): int  {.importcpp: "TORCH_ARG".}
  ## normalization module

proc TORCH_ARG*(this: var TransformerDecoderOptions): int  {.importcpp: "TORCH_ARG".}
  ## decoder layer to be cloned

proc TORCH_ARG*(this: var TransformerDecoderOptions): int  {.importcpp: "TORCH_ARG".}
  ## number of decoder layers

proc TORCH_ARG*(this: var TransformerDecoderOptions): int  {.importcpp: "TORCH_ARG".}
  ## normalization module

{.pop.} # header: "nn/options/transformercoder.h
