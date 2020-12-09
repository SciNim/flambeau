{.push header: "nn/options/pooling.h".}


# Constructors and methods
proc constructor_AdaptiveMaxPoolOptions<output_size_t>*(output_size: output_size_t): AdaptiveMaxPoolOptions {.constructor,importcpp: "AdaptiveMaxPoolOptions<output_size_t>(@)".}

proc constructor_AdaptiveAvgPoolOptions<output_size_t>*(output_size: output_size_t): AdaptiveAvgPoolOptions {.constructor,importcpp: "AdaptiveAvgPoolOptions<output_size_t>(@)".}

proc constructor_LPPoolOptions<D>*(norm_type: cdouble, kernel_size: cint): LPPoolOptions {.constructor,importcpp: "LPPoolOptions<D>(@)".}

proc stride_*(this: var AvgPoolOptions, cint): AvgPoolOptions<D>  {.importcpp: "stride_".}

proc TORCH_ARG*(this: var AvgPoolOptions): int  {.importcpp: "TORCH_ARG".}
  ## the stride of the window. Default value is `kernel_size`

proc TORCH_ARG*(this: var AvgPoolOptions): int  {.importcpp: "TORCH_ARG".}
  ## implicit zero padding to be added on both sides

proc TORCH_ARG*(this: var AvgPoolOptions): int  {.importcpp: "TORCH_ARG".}
  ## when True, will use `ceil` instead of `floor` to compute the output
  ## shape

proc TORCH_ARG*(this: var AvgPoolOptions): int  {.importcpp: "TORCH_ARG".}
  ## when True, will include the zero-padding in the averaging calculation

proc stride_*(this: var MaxPoolOptions, cint): MaxPoolOptions<D>  {.importcpp: "stride_".}

proc TORCH_ARG*(this: var MaxPoolOptions): int  {.importcpp: "TORCH_ARG".}
  ## the stride of the window. Default value is `kernel_size

proc TORCH_ARG*(this: var MaxPoolOptions): int  {.importcpp: "TORCH_ARG".}
  ## implicit zero padding to be added on both sides

proc TORCH_ARG*(this: var MaxPoolOptions): int  {.importcpp: "TORCH_ARG".}
  ## a parameter that controls the stride of elements in the window

proc TORCH_ARG*(this: var MaxPoolOptions): int  {.importcpp: "TORCH_ARG".}
  ## when True, will use `ceil` instead of `floor` to compute the output
  ## shape

proc TORCH_ARG*(this: var AdaptiveMaxPoolOptions): int  {.importcpp: "TORCH_ARG".}
  ## the target output size

proc TORCH_ARG*(this: var AdaptiveAvgPoolOptions): int  {.importcpp: "TORCH_ARG".}
  ## the target output size

proc stride_*(this: var MaxUnpoolOptions, cint): MaxUnpoolOptions<D>  {.importcpp: "stride_".}

proc TORCH_ARG*(this: var MaxUnpoolOptions): int  {.importcpp: "TORCH_ARG".}
  ## the stride of the window. Default value is `kernel_size

proc TORCH_ARG*(this: var MaxUnpoolOptions): int  {.importcpp: "TORCH_ARG".}
  ## implicit zero padding to be added on both sides

proc stride_*(this: var MaxUnpoolFuncOptions, cint): MaxUnpoolFuncOptions<D>  {.importcpp: "stride_".}

proc TORCH_ARG*(this: var MaxUnpoolFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## the stride of the window. Default value is `kernel_size

proc TORCH_ARG*(this: var MaxUnpoolFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## implicit zero padding to be added on both sides

proc TORCH_ARG*(this: var FractionalMaxPoolOptions): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var LPPoolOptions): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var LPPoolOptions): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var LPPoolOptions): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var LPPoolOptions): int  {.importcpp: "TORCH_ARG".}

{.pop.} # header: "nn/options/pooling.h
