{.push header: "nn/modules/conv.h".}


# Constructors and methods
proc constructor_ConvNdImpl<D, Derived>*(options_: cint): ConvNdImpl {.constructor,importcpp: "ConvNdImpl<D, Derived>(@)".}

proc constructor_Conv1dImpl*(input_channels: cint, output_channels: cint, kernel_size: cint): Conv1dImpl {.constructor,importcpp: "Conv1dImpl(@)".}

proc constructor_Conv1dImpl*(options_: cint): Conv1dImpl {.constructor,importcpp: "Conv1dImpl(@)".}

proc constructor_Conv2dImpl*(input_channels: cint, output_channels: cint, kernel_size: cint): Conv2dImpl {.constructor,importcpp: "Conv2dImpl(@)".}

proc constructor_Conv2dImpl*(options_: cint): Conv2dImpl {.constructor,importcpp: "Conv2dImpl(@)".}

proc constructor_Conv3dImpl*(input_channels: cint, output_channels: cint, kernel_size: cint): Conv3dImpl {.constructor,importcpp: "Conv3dImpl(@)".}

proc constructor_Conv3dImpl*(options_: cint): Conv3dImpl {.constructor,importcpp: "Conv3dImpl(@)".}

proc constructor_ConvTranspose1dImpl*(input_channels: cint, output_channels: cint, kernel_size: cint): ConvTranspose1dImpl {.constructor,importcpp: "ConvTranspose1dImpl(@)".}

proc constructor_ConvTranspose1dImpl*(options_: cint): ConvTranspose1dImpl {.constructor,importcpp: "ConvTranspose1dImpl(@)".}

proc constructor_ConvTranspose2dImpl*(input_channels: cint, output_channels: cint, kernel_size: cint): ConvTranspose2dImpl {.constructor,importcpp: "ConvTranspose2dImpl(@)".}

proc constructor_ConvTranspose2dImpl*(options_: cint): ConvTranspose2dImpl {.constructor,importcpp: "ConvTranspose2dImpl(@)".}

proc constructor_ConvTranspose3dImpl*(input_channels: cint, output_channels: cint, kernel_size: cint): ConvTranspose3dImpl {.constructor,importcpp: "ConvTranspose3dImpl(@)".}

proc constructor_ConvTranspose3dImpl*(options_: cint): ConvTranspose3dImpl {.constructor,importcpp: "ConvTranspose3dImpl(@)".}

proc reset*(this: var ConvNdImpl)  {.importcpp: "reset".}

proc reset_parameters*(this: var ConvNdImpl)  {.importcpp: "reset_parameters".}

proc pretty_print*(this: ConvNdImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `Conv{1,2,3}d` module into the given `stream`.

proc forward*(this: var Conv1dImpl): int  {.importcpp: "forward".}

proc forward*(this: var Conv2dImpl): int  {.importcpp: "forward".}

proc _conv_forward*(this: var Conv2dImpl): int  {.importcpp: "_conv_forward".}

proc forward*(this: var Conv3dImpl): int  {.importcpp: "forward".}

proc pretty_print*(this: ConvTransposeNdImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `ConvTranspose{1,2,3}d` module into the given
  ## `stream`.

proc _output_padding*(this: var ConvTransposeNdImpl): int  {.importcpp: "_output_padding".}

proc forward*(this: var ConvTranspose1dImpl): int  {.importcpp: "forward".}

proc FORWARD_HAS_DEFAULT_ARGS*(this: var ConvTranspose1dImpl): int  {.importcpp: "FORWARD_HAS_DEFAULT_ARGS".}

proc forward*(this: var ConvTranspose2dImpl): int  {.importcpp: "forward".}

proc FORWARD_HAS_DEFAULT_ARGS*(this: var ConvTranspose2dImpl): int  {.importcpp: "FORWARD_HAS_DEFAULT_ARGS".}

proc forward*(this: var ConvTranspose3dImpl): int  {.importcpp: "forward".}

proc FORWARD_HAS_DEFAULT_ARGS*(this: var ConvTranspose3dImpl): int  {.importcpp: "FORWARD_HAS_DEFAULT_ARGS".}

{.pop.} # header: "nn/modules/conv.h
