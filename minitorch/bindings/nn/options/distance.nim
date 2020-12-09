{.push header: "nn/options/distance.h".}


# Constructors and methods
proc TORCH_ARG*(this: var CosineSimilarityOptions): int  {.importcpp: "TORCH_ARG".}
  ## Dimension where cosine similarity is computed. Default: 1

proc TORCH_ARG*(this: var CosineSimilarityOptions): int  {.importcpp: "TORCH_ARG".}
  ## Small value to avoid division by zero. Default: 1e-8

proc TORCH_ARG*(this: var PairwiseDistanceOptions): int  {.importcpp: "TORCH_ARG".}
  ## The norm degree. Default: 2

proc TORCH_ARG*(this: var PairwiseDistanceOptions): int  {.importcpp: "TORCH_ARG".}
  ## Small value to avoid division by zero. Default: 1e-6

proc TORCH_ARG*(this: var PairwiseDistanceOptions): int  {.importcpp: "TORCH_ARG".}
  ## Determines whether or not to keep the vector dimension. Default: false

{.pop.} # header: "nn/options/distance.h
