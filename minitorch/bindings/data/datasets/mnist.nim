{.push header: "data/datasets/mnist.h".}


# Constructors and methods
proc constructor_MNIST*(root: std::string, mode: torch::data::datasets::MNIST::Mode): MNIST {.constructor,importcpp: "MNIST(@)".}
  ## Loads the MNIST dataset from the `root` path.

proc get*(this: var MNIST): int  {.importcpp: "get".}
  ## Returns the `Example` at the given `index`.

proc size*(this: MNIST): int  {.importcpp: "size".}
  ## Returns the size of the dataset.

proc is_train*(this: MNIST): bool  {.importcpp: "is_train".}
  ## Returns true if this is the training subset of MNIST.

proc images*(this: MNIST): int  {.importcpp: "images".}
  ## Returns all images stacked into a single tensor.

proc targets*(this: MNIST): int  {.importcpp: "targets".}
  ## Returns all targets stacked into a single tensor.

{.pop.} # header: "data/datasets/mnist.h
