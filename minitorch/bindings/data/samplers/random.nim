{.push header: "data/samplers/random.h".}


# Constructors and methods
proc constructor_RandomSampler*(size: cint, index_dtype: cint): RandomSampler {.constructor,importcpp: "RandomSampler(@)".}
  ## Constructs a `RandomSampler` with a size and dtype for the stored
  ## indices.

proc reset*(this: var RandomSampler, new_size: cint)  {.importcpp: "reset".}
  ## Resets the `RandomSampler` to a new set of indices.

proc next*(this: var RandomSampler): int  {.importcpp: "next".}
  ## Returns the next batch of indices.

proc save*(this: RandomSampler, archive: var serialize::OutputArchive)  {.importcpp: "save".}
  ## Serializes the `RandomSampler` to the `archive`.

proc load*(this: var RandomSampler, archive: var serialize::InputArchive)  {.importcpp: "load".}
  ## Deserializes the `RandomSampler` from the `archive`.

proc index*(this: RandomSampler): int  {.importcpp: "index".}
  ## Returns the current index of the `RandomSampler`.

{.pop.} # header: "data/samplers/random.h
