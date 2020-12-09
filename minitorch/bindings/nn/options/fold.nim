{.push header: "nn/options/fold.h".}


# Constructors and methods
proc kernel_size_*(this: var FoldOptions, kernel_size: cint): torch::nn::FoldOptions  {.importcpp: "kernel_size_".}

proc TORCH_ARG*(this: var FoldOptions): int  {.importcpp: "TORCH_ARG".}
  ## the size of the sliding blocks

proc TORCH_ARG*(this: var FoldOptions): int  {.importcpp: "TORCH_ARG".}
  ## controls the spacing between the kernel points; also known as the à
  ## trous algorithm.

proc TORCH_ARG*(this: var FoldOptions): int  {.importcpp: "TORCH_ARG".}
  ## controls the amount of implicit zero-paddings on both sides for
  ## padding number of points for each dimension before reshaping.

proc TORCH_ARG*(this: var FoldOptions): int  {.importcpp: "TORCH_ARG".}
  ## controls the stride for the sliding blocks.

proc TORCH_ARG*(this: var UnfoldOptions): int  {.importcpp: "TORCH_ARG".}
  ## controls the spacing between the kernel points; also known as the à
  ## trous algorithm.

proc TORCH_ARG*(this: var UnfoldOptions): int  {.importcpp: "TORCH_ARG".}
  ## controls the amount of implicit zero-paddings on both sides for
  ## padding number of points for each dimension before reshaping.

proc TORCH_ARG*(this: var UnfoldOptions): int  {.importcpp: "TORCH_ARG".}
  ## controls the stride for the sliding blocks.

{.pop.} # header: "nn/options/fold.h
