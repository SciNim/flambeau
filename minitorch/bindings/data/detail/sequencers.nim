{.push header: "data/detail/sequencers.h".}


# Constructors and methods
proc constructor_OrderedSequencer<Result>*(max_jobs: cint): OrderedSequencer {.constructor,importcpp: "OrderedSequencer<Result>(@)".}
  ## Constructs the `OrderedSequencer` with the maximum number of results
  ## it will ever hold at one point in time.

proc next*(this: var Sequencer): int  {.importcpp: "next".}

proc next*(this: var NoSequencer): int  {.importcpp: "next".}

proc next*(this: var OrderedSequencer): int  {.importcpp: "next".}
  ## Buffers results until the next one in the expected order is received.

proc buffer*(this: var OrderedSequencer): int  {.importcpp: "buffer".}
  ## Accesses the buffer at the `index` modulo the buffer size.

{.pop.} # header: "data/detail/sequencers.h
