{.push header: "data/datasets/base.h".}


# Constructors and methods
proc get_batch*(this: var BatchDataset, request: BatchRequest): Batch  {.importcpp: "get_batch".}
  ## Returns a batch of data given an index.

proc size*(this: BatchDataset): int  {.importcpp: "size".}
  ## Returns the size of the dataset, or an empty optional if it is
  ## unsized.

proc get*(this: var Dataset, index: cint): torch::data::datasets::Dataset::ExampleType  {.importcpp: "get".}
  ## Returns the example at the given index.

proc get_batch*(this: var Dataset): int  {.importcpp: "get_batch".}
  ## Returns a batch of data. The default implementation calls `get()` for
  ## every requested index in the batch.

{.pop.} # header: "data/datasets/base.h
