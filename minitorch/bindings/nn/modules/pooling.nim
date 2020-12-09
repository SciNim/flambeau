{.push header: "nn/modules/pooling.h".}


# Constructors and methods
proc constructor_AdaptiveMaxPoolImpl<D, output_size_t, Derived>*(output_size: output_size_t): AdaptiveMaxPoolImpl {.constructor,importcpp: "AdaptiveMaxPoolImpl<D, output_size_t, Derived>(@)".}

proc constructor_AdaptiveMaxPoolImpl<D, output_size_t, Derived>*(options_: cint): AdaptiveMaxPoolImpl {.constructor,importcpp: "AdaptiveMaxPoolImpl<D, output_size_t, Derived>(@)".}

proc constructor_AdaptiveAvgPoolImpl<D, output_size_t, Derived>*(output_size: output_size_t): AdaptiveAvgPoolImpl {.constructor,importcpp: "AdaptiveAvgPoolImpl<D, output_size_t, Derived>(@)".}

proc constructor_AdaptiveAvgPoolImpl<D, output_size_t, Derived>*(options_: cint): AdaptiveAvgPoolImpl {.constructor,importcpp: "AdaptiveAvgPoolImpl<D, output_size_t, Derived>(@)".}

proc constructor_LPPoolImpl<D, Derived>*(norm_type: cdouble, kernel_size: cint): LPPoolImpl {.constructor,importcpp: "LPPoolImpl<D, Derived>(@)".}

proc constructor_LPPoolImpl<D, Derived>*(options_: cint): LPPoolImpl {.constructor,importcpp: "LPPoolImpl<D, Derived>(@)".}

proc reset*(this: var AvgPoolImpl)  {.importcpp: "reset".}

proc pretty_print*(this: AvgPoolImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `AvgPool{1,2,3}d` module into the given `stream`.

proc forward*(this: var AvgPool1dImpl): int  {.importcpp: "forward".}

proc forward*(this: var AvgPool2dImpl): int  {.importcpp: "forward".}

proc forward*(this: var AvgPool3dImpl): int  {.importcpp: "forward".}

proc reset*(this: var MaxPoolImpl)  {.importcpp: "reset".}

proc pretty_print*(this: MaxPoolImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `MaxPool{1,2,3}d` module into the given `stream`.

proc forward*(this: var MaxPool1dImpl): int  {.importcpp: "forward".}

proc forward_with_indices*(this: var MaxPool1dImpl): int  {.importcpp: "forward_with_indices".}
  ## Returns the outputs and the indices of the max values. Useful for
  ## `torch::nn::MaxUnpool1d` later.

proc forward*(this: var MaxPool2dImpl): int  {.importcpp: "forward".}

proc forward_with_indices*(this: var MaxPool2dImpl): int  {.importcpp: "forward_with_indices".}
  ## Returns the outputs and the indices of the max values. Useful for
  ## `torch::nn::MaxUnpool2d` later.

proc forward*(this: var MaxPool3dImpl): int  {.importcpp: "forward".}

proc forward_with_indices*(this: var MaxPool3dImpl): int  {.importcpp: "forward_with_indices".}
  ## Returns the outputs and the indices of the max values. Useful for
  ## `torch::nn::MaxUnpool3d` later.

proc reset*(this: var AdaptiveMaxPoolImpl)  {.importcpp: "reset".}

proc pretty_print*(this: AdaptiveMaxPoolImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `AdaptiveMaxPool{1,2,3}d` module into the given
  ## `stream`.

proc forward*(this: var AdaptiveMaxPool1dImpl): int  {.importcpp: "forward".}

proc forward_with_indices*(this: var AdaptiveMaxPool1dImpl): int  {.importcpp: "forward_with_indices".}
  ## Returns the indices along with the outputs. Useful to pass to
  ## nn.MaxUnpool1d.

proc forward*(this: var AdaptiveMaxPool2dImpl): int  {.importcpp: "forward".}

proc forward_with_indices*(this: var AdaptiveMaxPool2dImpl): int  {.importcpp: "forward_with_indices".}
  ## Returns the indices along with the outputs. Useful to pass to
  ## nn.MaxUnpool2d.

proc forward*(this: var AdaptiveMaxPool3dImpl): int  {.importcpp: "forward".}

proc forward_with_indices*(this: var AdaptiveMaxPool3dImpl): int  {.importcpp: "forward_with_indices".}
  ## Returns the indices along with the outputs. Useful to pass to
  ## nn.MaxUnpool3d.

proc reset*(this: var AdaptiveAvgPoolImpl)  {.importcpp: "reset".}

proc pretty_print*(this: AdaptiveAvgPoolImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `AdaptiveAvgPool{1,2,3}d` module into the given
  ## `stream`.

proc forward*(this: var AdaptiveAvgPool1dImpl): int  {.importcpp: "forward".}

proc forward*(this: var AdaptiveAvgPool2dImpl): int  {.importcpp: "forward".}

proc forward*(this: var AdaptiveAvgPool3dImpl): int  {.importcpp: "forward".}

proc reset*(this: var MaxUnpoolImpl)  {.importcpp: "reset".}

proc pretty_print*(this: MaxUnpoolImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `MaxUnpool{1,2,3}d` module into the given `stream`.

proc forward*(this: var MaxUnpool1dImpl): int  {.importcpp: "forward".}

proc FORWARD_HAS_DEFAULT_ARGS*(this: var MaxUnpool1dImpl): int  {.importcpp: "FORWARD_HAS_DEFAULT_ARGS".}

proc forward*(this: var MaxUnpool2dImpl): int  {.importcpp: "forward".}

proc FORWARD_HAS_DEFAULT_ARGS*(this: var MaxUnpool2dImpl): int  {.importcpp: "FORWARD_HAS_DEFAULT_ARGS".}

proc forward*(this: var MaxUnpool3dImpl): int  {.importcpp: "forward".}

proc FORWARD_HAS_DEFAULT_ARGS*(this: var MaxUnpool3dImpl): int  {.importcpp: "FORWARD_HAS_DEFAULT_ARGS".}

proc reset*(this: var FractionalMaxPool2dImpl)  {.importcpp: "reset".}

proc pretty_print*(this: FractionalMaxPool2dImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `FractionalMaxPool2d` module into the given
  ## `stream`.

proc forward*(this: var FractionalMaxPool2dImpl): int  {.importcpp: "forward".}

proc forward_with_indices*(this: var FractionalMaxPool2dImpl): int  {.importcpp: "forward_with_indices".}
  ## Returns the outputs and the indices of the max values. Useful for
  ## `torch::nn::MaxUnpool2d` later.

proc reset*(this: var FractionalMaxPool3dImpl)  {.importcpp: "reset".}

proc pretty_print*(this: FractionalMaxPool3dImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `FractionalMaxPool3d` module into the given
  ## `stream`.

proc forward*(this: var FractionalMaxPool3dImpl): int  {.importcpp: "forward".}

proc forward_with_indices*(this: var FractionalMaxPool3dImpl): int  {.importcpp: "forward_with_indices".}
  ## Returns the outputs and the indices of the max values. Useful for
  ## `torch::nn::MaxUnpool3d` later.

proc reset*(this: var LPPoolImpl)  {.importcpp: "reset".}

proc pretty_print*(this: LPPoolImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `LPPool{1,2}d` module into the given `stream`.

proc forward*(this: var LPPool1dImpl): int  {.importcpp: "forward".}

proc forward*(this: var LPPool2dImpl): int  {.importcpp: "forward".}

{.pop.} # header: "nn/modules/pooling.h
