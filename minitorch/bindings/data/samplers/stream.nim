{.push header: "data/samplers/stream.h".}


# Constructors and methods
proc constructor_BatchSize*(size: cint): BatchSize {.constructor,importcpp: "BatchSize(@)".}

proc constructor_StreamSampler*(epoch_size: cint): StreamSampler {.constructor,importcpp: "StreamSampler(@)".}
  ## Constructs the `StreamSampler` with the number of individual examples
  ## that should be fetched until the sampler is exhausted.

proc size*(this: BatchSize): int  {.importcpp: "size".}

proc reset*(this: var StreamSampler, new_size: cint)  {.importcpp: "reset".}
  ## Resets the internal state of the sampler.

proc next*(this: var StreamSampler): int  {.importcpp: "next".}
  ## Returns a `BatchSize` object with the number of elements to fetch in
  ## the next batch. This number is the minimum of the supplied
  ## `batch_size` and the difference between the `epoch_size` and the
  ## current index. If the `epoch_size` has been reached, returns an empty
  ## optional.

proc save*(this: StreamSampler, archive: var serialize::OutputArchive)  {.importcpp: "save".}
  ## Serializes the `StreamSampler` to the `archive`.

proc load*(this: var StreamSampler, archive: var serialize::InputArchive)  {.importcpp: "load".}
  ## Deserializes the `StreamSampler` from the `archive`.

{.pop.} # header: "data/samplers/stream.h
