{.push header: "data/samplers/base.h".}


# Constructors and methods
proc reset*(this: var Sampler, new_size: cint)  {.importcpp: "reset".}
  ## Resets the `Sampler`'s internal state. Typically called before a new
  ## epoch. Optionally, accepts a new size when reseting the sampler.

proc next*(this: var Sampler): int  {.importcpp: "next".}
  ## Returns the next index if possible, or an empty optional if the
  ## sampler is exhausted for this epoch.

proc save*(this: Sampler, archive: var serialize::OutputArchive)  {.importcpp: "save".}
  ## Serializes the `Sampler` to the `archive`.

proc load*(this: var Sampler, archive: var serialize::InputArchive)  {.importcpp: "load".}
  ## Deserializes the `Sampler` from the `archive`.

{.pop.} # header: "data/samplers/base.h
