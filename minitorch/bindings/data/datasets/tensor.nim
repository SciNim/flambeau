{.push header: "data/datasets/tensor.h".}


# Constructors and methods
proc constructor_TensorDataset*(tensors: cint): TensorDataset {.constructor,importcpp: "TensorDataset(@)".}
  ## Creates a `TensorDataset` from a vector of tensors.

proc constructor_TensorDataset*(tensor: cint): TensorDataset {.constructor,importcpp: "TensorDataset(@)".}

proc get*(this: var TensorDataset): int  {.importcpp: "get".}
  ## Returns a single `TensorExample`.

proc size*(this: TensorDataset): int  {.importcpp: "size".}
  ## Returns the number of tensors in the dataset.

{.pop.} # header: "data/datasets/tensor.h
