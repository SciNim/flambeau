{.push header: "data/samplers/sequential.h".}


# Constructors and methods
proc constructor_SequentialSampler*(size: cint): SequentialSampler {.constructor,importcpp: "SequentialSampler(@)".}
  ## Creates a `SequentialSampler` that will return indices in the range
  ## `0...size - 1`.

proc reset*(this: var SequentialSampler, new_size: cint)  {.importcpp: "reset".}
  ## Resets the `SequentialSampler` to zero.

proc next*(this: var SequentialSampler): int  {.importcpp: "next".}
  ## Returns the next batch of indices.

proc save*(this: SequentialSampler, archive: var serialize::OutputArchive)  {.importcpp: "save".}
  ## Serializes the `SequentialSampler` to the `archive`.

proc load*(this: var SequentialSampler, archive: var serialize::InputArchive)  {.importcpp: "load".}
  ## Deserializes the `SequentialSampler` from the `archive`.

proc index*(this: SequentialSampler): int  {.importcpp: "index".}
  ## Returns the current index of the `SequentialSampler`.

{.pop.} # header: "data/samplers/sequential.h
