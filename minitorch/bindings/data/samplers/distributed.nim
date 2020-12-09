{.push header: "data/samplers/distributed.h".}


# Constructors and methods
proc constructor_DistributedSampler<BatchRequest>*(size: cint, num_replicas: cint, rank: cint, allow_duplicates: bool): DistributedSampler {.constructor,importcpp: "DistributedSampler<BatchRequest>(@)".}

proc constructor_DistributedRandomSampler*(size: cint, num_replicas: cint, rank: cint, allow_duplicates: bool): DistributedRandomSampler {.constructor,importcpp: "DistributedRandomSampler(@)".}

proc constructor_DistributedSequentialSampler*(size: cint, num_replicas: cint, rank: cint, allow_duplicates: bool): DistributedSequentialSampler {.constructor,importcpp: "DistributedSequentialSampler(@)".}

proc set_epoch*(this: var DistributedSampler, epoch: cint)  {.importcpp: "set_epoch".}
  ## Set the epoch for the current enumeration. This can be used to alter
  ## the sample selection and shuffling behavior.

proc epoch*(this: DistributedSampler): int  {.importcpp: "epoch".}

proc local_sample_count*(this: var DistributedSampler): int  {.importcpp: "local_sample_count".}

proc reset*(this: var DistributedRandomSampler, new_size: cint)  {.importcpp: "reset".}
  ## Resets the `DistributedRandomSampler` to a new set of indices.

proc next*(this: var DistributedRandomSampler): int  {.importcpp: "next".}
  ## Returns the next batch of indices.

proc save*(this: DistributedRandomSampler, archive: var serialize::OutputArchive)  {.importcpp: "save".}
  ## Serializes the `DistributedRandomSampler` to the `archive`.

proc load*(this: var DistributedRandomSampler, archive: var serialize::InputArchive)  {.importcpp: "load".}
  ## Deserializes the `DistributedRandomSampler` from the `archive`.

proc index*(this: DistributedRandomSampler): int  {.importcpp: "index".}
  ## Returns the current index of the `DistributedRandomSampler`.

proc populate_indices*(this: var DistributedRandomSampler)  {.importcpp: "populate_indices".}

proc reset*(this: var DistributedSequentialSampler, new_size: cint)  {.importcpp: "reset".}
  ## Resets the `DistributedSequentialSampler` to a new set of indices.

proc next*(this: var DistributedSequentialSampler): int  {.importcpp: "next".}
  ## Returns the next batch of indices.

proc save*(this: DistributedSequentialSampler, archive: var serialize::OutputArchive)  {.importcpp: "save".}
  ## Serializes the `DistributedSequentialSampler` to the `archive`.

proc load*(this: var DistributedSequentialSampler, archive: var serialize::InputArchive)  {.importcpp: "load".}
  ## Deserializes the `DistributedSequentialSampler` from the `archive`.

proc index*(this: DistributedSequentialSampler): int  {.importcpp: "index".}
  ## Returns the current index of the `DistributedSequentialSampler`.

proc populate_indices*(this: var DistributedSequentialSampler)  {.importcpp: "populate_indices".}

{.pop.} # header: "data/samplers/distributed.h
