{.push header: "data/datasets/shared.h".}


# Constructors and methods
proc constructor_SharedBatchDataset<UnderlyingDataset>*(shared_dataset: std::shared_ptr<UnderlyingDataset>): SharedBatchDataset {.constructor,importcpp: "SharedBatchDataset<UnderlyingDataset>(@)".}
  ## Constructs a new `SharedBatchDataset` from a `shared_ptr` to the
  ## `UnderlyingDataset`.

proc get_batch*(this: var SharedBatchDataset, request: torch::data::datasets::SharedBatchDataset::BatchRequestType): torch::data::datasets::SharedBatchDataset::BatchType  {.importcpp: "get_batch".}
  ## Calls `get_batch` on the underlying dataset.

proc size*(this: SharedBatchDataset): int  {.importcpp: "size".}
  ## Returns the `size` from the underlying dataset.

proc `*`*(this: var SharedBatchDataset): UnderlyingDataset  {.importcpp: "`*`".}
  ## Accesses the underlying dataset.

proc `*`*(this: SharedBatchDataset): UnderlyingDataset  {.importcpp: "`*`".}
  ## Accesses the underlying dataset.

proc `->`*(this: var SharedBatchDataset): UnderlyingDataset *  {.importcpp: "`->`".}
  ## Accesses the underlying dataset.

proc `->`*(this: SharedBatchDataset): UnderlyingDataset *  {.importcpp: "`->`".}
  ## Accesses the underlying dataset.

proc reset*(this: var SharedBatchDataset)  {.importcpp: "reset".}
  ## Calls `reset()` on the underlying dataset.

{.pop.} # header: "data/datasets/shared.h
