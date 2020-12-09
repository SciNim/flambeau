{.push header: "nn/options/upsampling.h".}


# Constructors and methods
proc TORCH_ARG*(this: var UpsampleOptions): int  {.importcpp: "TORCH_ARG".}
  ## output spatial sizes.

proc TORCH_ARG*(this: var UpsampleOptions): int  {.importcpp: "TORCH_ARG".}
  ## multiplier for spatial size.

proc TORCH_ARG*(this: var UpsampleOptions): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var UpsampleOptions): int  {.importcpp: "TORCH_ARG".}
  ## if "True", the corner pixels of the input and output tensors are
  ## aligned, and thus preserving the values at those pixels. This only has
  ## effect when :attr:`mode` is "linear", "bilinear", or "trilinear".
  ## Default: "False"

proc TORCH_ARG*(this: var InterpolateFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## output spatial sizes.

proc TORCH_ARG*(this: var InterpolateFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## multiplier for spatial size.

proc TORCH_ARG*(this: var InterpolateFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## the upsampling algorithm: one of "nearest", "linear", "bilinear",
  ## "bicubic", "trilinear", and "area". Default: "nearest"

proc TORCH_ARG*(this: var InterpolateFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## Geometrically, we consider the pixels of the input and output as
  ## squares rather than points. If set to "True", the input and output
  ## tensors are aligned by the center points of their corner pixels,
  ## preserving the values at the corner pixels. If set to "False", the
  ## input and output tensors are aligned by the corner points of their
  ## corner pixels, and the interpolation uses edge value padding for out-
  ## of-boundary values, making this operation *independent* of input size
  ## when :attr:`scale_factor` is kept the same. This only has an effect
  ## when :attr:`mode` is "linear", "bilinear", "bicubic" or "trilinear".
  ## Default: "False"

proc TORCH_ARG*(this: var InterpolateFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## recompute the scale_factor for use in the interpolation calculation.
  ## When `scale_factor` is passed as a parameter, it is used to compute
  ## the `output_size`. If `recompute_scale_factor` is `true` or not
  ## specified, a new `scale_factor` will be computed based on the output
  ## and input sizes for use in the interpolation computation (i.e. the
  ## computation will be identical to if the computed `output_size` were
  ## passed-in explicitly). Otherwise, the passed-in `scale_factor` will be
  ## used in the interpolation computation. Note that when `scale_factor`
  ## is floating-point, the recomputed scale_factor may differ from the one
  ## passed in due to rounding and precision issues.

{.pop.} # header: "nn/options/upsampling.h
