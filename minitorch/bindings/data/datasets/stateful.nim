{.push header: "data/datasets/stateful.h".}


# Constructors and methods
proc reset*(this: var StatefulDataset)  {.importcpp: "reset".}
  ## Resets internal state of the dataset.

proc save*(this: StatefulDataset, archive: var serialize::OutputArchive)  {.importcpp: "save".}
  ## Saves the statefulDataset's state to OutputArchive.

proc load*(this: var StatefulDataset, archive: var serialize::InputArchive)  {.importcpp: "load".}
  ## Deserializes the statefulDataset's state from the `archive`.

{.pop.} # header: "data/datasets/stateful.h
