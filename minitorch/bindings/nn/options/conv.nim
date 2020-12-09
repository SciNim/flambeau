{.push header: "nn/options/conv.h".}


# Constructors and methods
proc constructor_ConvNdOptions<D>*(in_channels: cint, out_channels: cint, kernel_size: cint): ConvNdOptions {.constructor,importcpp: "ConvNdOptions<D>(@)".}

proc constructor_ConvOptions<D>*(in_channels: cint, out_channels: cint, kernel_size: cint): ConvOptions {.constructor,importcpp: "ConvOptions<D>(@)".}

proc constructor_ConvTransposeOptions<D>*(in_channels: cint, out_channels: cint, kernel_size: cint): ConvTransposeOptions {.constructor,importcpp: "ConvTransposeOptions<D>(@)".}

proc TORCH_ARG*(this: var ConvNdOptions): int  {.importcpp: "TORCH_ARG".}
  ## The number of channels the input volumes will have. Changing this
  ## parameter after construction __has no effect__.

proc TORCH_ARG*(this: var ConvNdOptions): int  {.importcpp: "TORCH_ARG".}
  ## The number of output channels the convolution should produce. Changing
  ## this parameter after construction __has no effect__.

proc TORCH_ARG*(this: var ConvNdOptions): int  {.importcpp: "TORCH_ARG".}
  ## The kernel size to use. For a `D`-dim convolution, must be a single
  ## number or a list of `D` numbers. This parameter __can__ be changed
  ## after construction.

proc TORCH_ARG*(this: var ConvNdOptions): int  {.importcpp: "TORCH_ARG".}
  ## The stride of the convolution. For a `D`-dim convolution, must be a
  ## single number or a list of `D` numbers. This parameter __can__ be
  ## changed after construction.

proc TORCH_ARG*(this: var ConvNdOptions): int  {.importcpp: "TORCH_ARG".}
  ## The padding to add to the input volumes. For a `D`-dim convolution,
  ## must be a single number or a list of `D` numbers. This parameter
  ## __can__ be changed after construction.

proc TORCH_ARG*(this: var ConvNdOptions): int  {.importcpp: "TORCH_ARG".}
  ## The kernel dilation. For a `D`-dim convolution, must be a single
  ## number or a list of `D` numbers. This parameter __can__ be changed
  ## after construction.

proc TORCH_ARG*(this: var ConvNdOptions): int  {.importcpp: "TORCH_ARG".}
  ## If true, convolutions will be transpose convolutions (a.k.a.
  ## deconvolutions). Changing this parameter after construction __has no
  ## effect__.

proc TORCH_ARG*(this: var ConvNdOptions): int  {.importcpp: "TORCH_ARG".}
  ## For transpose convolutions, the padding to add to output volumes. For
  ## a `D`-dim convolution, must be a single number or a list of `D`
  ## numbers. This parameter __can__ be changed after construction.

proc TORCH_ARG*(this: var ConvNdOptions): int  {.importcpp: "TORCH_ARG".}
  ## The number of convolution groups. This parameter __can__ be changed
  ## after construction.

proc TORCH_ARG*(this: var ConvNdOptions): int  {.importcpp: "TORCH_ARG".}
  ## Whether to add a bias after individual applications of the kernel.
  ## Changing this parameter after construction __has no effect__.

proc TORCH_ARG*(this: var ConvNdOptions): int  {.importcpp: "TORCH_ARG".}
  ## Accepted values `torch::kZeros`, `torch::kReflect`,
  ## `torch::kReplicate` or `torch::kCircular`. Default: `torch::kZeros`

proc TORCH_ARG*(this: var ConvOptions): int  {.importcpp: "TORCH_ARG".}
  ## The number of channels the input volumes will have. Changing this
  ## parameter after construction __has no effect__.

proc TORCH_ARG*(this: var ConvOptions): int  {.importcpp: "TORCH_ARG".}
  ## The number of output channels the convolution should produce. Changing
  ## this parameter after construction __has no effect__.

proc TORCH_ARG*(this: var ConvOptions): int  {.importcpp: "TORCH_ARG".}
  ## The kernel size to use. For a `D`-dim convolution, must be a single
  ## number or a list of `D` numbers. This parameter __can__ be changed
  ## after construction.

proc TORCH_ARG*(this: var ConvOptions): int  {.importcpp: "TORCH_ARG".}
  ## The stride of the convolution. For a `D`-dim convolution, must be a
  ## single number or a list of `D` numbers. This parameter __can__ be
  ## changed after construction.

proc TORCH_ARG*(this: var ConvOptions): int  {.importcpp: "TORCH_ARG".}
  ## The padding to add to the input volumes. For a `D`-dim convolution,
  ## must be a single number or a list of `D` numbers. This parameter
  ## __can__ be changed after construction.

proc TORCH_ARG*(this: var ConvOptions): int  {.importcpp: "TORCH_ARG".}
  ## The kernel dilation. For a `D`-dim convolution, must be a single
  ## number or a list of `D` numbers. This parameter __can__ be changed
  ## after construction.

proc TORCH_ARG*(this: var ConvOptions): int  {.importcpp: "TORCH_ARG".}
  ## The number of convolution groups. This parameter __can__ be changed
  ## after construction.

proc TORCH_ARG*(this: var ConvOptions): int  {.importcpp: "TORCH_ARG".}
  ## Whether to add a bias after individual applications of the kernel.
  ## Changing this parameter after construction __has no effect__.

proc TORCH_ARG*(this: var ConvOptions): int  {.importcpp: "TORCH_ARG".}
  ## Accepted values `torch::kZeros`, `torch::kReflect`,
  ## `torch::kReplicate` or `torch::kCircular`. Default: `torch::kZeros`

proc TORCH_ARG*(this: var ConvFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## optional bias of shape `(out_channels)`. Default: ``None``

proc TORCH_ARG*(this: var ConvFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## The stride of the convolving kernel. For a `D`-dim convolution, must
  ## be a single number or a list of `D` numbers.

proc TORCH_ARG*(this: var ConvFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## Implicit paddings on both sides of the input. For a `D`-dim
  ## convolution, must be a single number or a list of `D` numbers.

proc TORCH_ARG*(this: var ConvFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## The spacing between kernel elements. For a `D`-dim convolution, must
  ## be a single number or a list of `D` numbers.

proc TORCH_ARG*(this: var ConvFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## Split input into groups, `in_channels` should be divisible by the
  ## number of groups.

proc TORCH_ARG*(this: var ConvTransposeOptions): int  {.importcpp: "TORCH_ARG".}
  ## The number of channels the input volumes will have. Changing this
  ## parameter after construction __has no effect__.

proc TORCH_ARG*(this: var ConvTransposeOptions): int  {.importcpp: "TORCH_ARG".}
  ## The number of output channels the convolution should produce. Changing
  ## this parameter after construction __has no effect__.

proc TORCH_ARG*(this: var ConvTransposeOptions): int  {.importcpp: "TORCH_ARG".}
  ## The kernel size to use. For a `D`-dim convolution, must be a single
  ## number or a list of `D` numbers. This parameter __can__ be changed
  ## after construction.

proc TORCH_ARG*(this: var ConvTransposeOptions): int  {.importcpp: "TORCH_ARG".}
  ## The stride of the convolution. For a `D`-dim convolution, must be a
  ## single number or a list of `D` numbers. This parameter __can__ be
  ## changed after construction.

proc TORCH_ARG*(this: var ConvTransposeOptions): int  {.importcpp: "TORCH_ARG".}
  ## The padding to add to the input volumes. For a `D`-dim convolution,
  ## must be a single number or a list of `D` numbers. This parameter
  ## __can__ be changed after construction.

proc TORCH_ARG*(this: var ConvTransposeOptions): int  {.importcpp: "TORCH_ARG".}
  ## For transpose convolutions, the padding to add to output volumes. For
  ## a `D`-dim convolution, must be a single number or a list of `D`
  ## numbers. This parameter __can__ be changed after construction.

proc TORCH_ARG*(this: var ConvTransposeOptions): int  {.importcpp: "TORCH_ARG".}
  ## The number of convolution groups. This parameter __can__ be changed
  ## after construction.

proc TORCH_ARG*(this: var ConvTransposeOptions): int  {.importcpp: "TORCH_ARG".}
  ## Whether to add a bias after individual applications of the kernel.
  ## Changing this parameter after construction __has no effect__.

proc TORCH_ARG*(this: var ConvTransposeOptions): int  {.importcpp: "TORCH_ARG".}
  ## The kernel dilation. For a `D`-dim convolution, must be a single
  ## number or a list of `D` numbers. This parameter __can__ be changed
  ## after construction.

proc TORCH_ARG*(this: var ConvTransposeOptions): int  {.importcpp: "TORCH_ARG".}
  ## Accepted values `torch::kZeros`, `torch::kReflect`,
  ## `torch::kReplicate` or `torch::kCircular`. Default: `torch::kZeros`

proc TORCH_ARG*(this: var ConvTransposeFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## optional bias of shape `(out_channels)`. Default: ``None``

proc TORCH_ARG*(this: var ConvTransposeFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## The stride of the convolving kernel. For a `D`-dim convolution, must
  ## be a single number or a list of `D` numbers.

proc TORCH_ARG*(this: var ConvTransposeFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## Implicit paddings on both sides of the input. For a `D`-dim
  ## convolution, must be a single number or a list of `D` numbers.

proc TORCH_ARG*(this: var ConvTransposeFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## Additional size added to one side of each dimension in the output
  ## shape. Default: 0

proc TORCH_ARG*(this: var ConvTransposeFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## Split input into groups, `in_channels` should be divisible by the
  ## number of groups.

proc TORCH_ARG*(this: var ConvTransposeFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## The spacing between kernel elements. For a `D`-dim convolution, must
  ## be a single number or a list of `D` numbers.

{.pop.} # header: "nn/options/conv.h
