{.push header: "data/dataloader/stateful.h".}


# Constructors and methods
proc constructor_StatefulDataLoader<Dataset>*(dataset: Dataset, options: cint): StatefulDataLoader {.constructor,importcpp: "StatefulDataLoader<Dataset>(@)".}
  ## Constructs the `StatefulDataLoader` from a `dataset` and some
  ## `options`.

proc reset*(this: var StatefulDataLoader)  {.importcpp: "reset".}
  ## Resets the internal state of the dataloader and the dataset.

proc get_batch_request*(this: var StatefulDataLoader): int  {.importcpp: "get_batch_request".}
  ## For stateful datasets, the batch request is always the batch size. The
  ## dataset is responsible for determining what goes into the batch next.

{.pop.} # header: "data/dataloader/stateful.h
