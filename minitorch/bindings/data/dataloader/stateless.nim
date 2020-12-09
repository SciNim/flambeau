{.push header: "data/dataloader/stateless.h".}


# Constructors and methods
proc constructor_StatelessDataLoader<Dataset, Sampler>*(dataset: Dataset, sampler: Sampler, options: cint): StatelessDataLoader {.constructor,importcpp: "StatelessDataLoader<Dataset, Sampler>(@)".}
  ## Constructs the `StatelessDataLoader` from a `dataset`, a `sampler` and
  ## some `options`.

proc reset*(this: var StatelessDataLoader)  {.importcpp: "reset".}
  ## Resets the internal state of the dataloader and the sampler.

proc get_batch_request*(this: var StatelessDataLoader): int  {.importcpp: "get_batch_request".}
  ## Queries the sampler for the next batch request (possibly progressing
  ## its internal state).

{.pop.} # header: "data/dataloader/stateless.h
