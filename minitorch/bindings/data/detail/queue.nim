{.push header: "data/detail/queue.h".}


# Constructors and methods
proc push*(this: var Queue, value: T)  {.importcpp: "push".}
  ## Pushes a new value to the back of the `Queue` and notifies one thread
  ## on the waiting side about this event.

proc pop*(this: var Queue, timeout: cint): T  {.importcpp: "pop".}
  ## Blocks until at least one element is ready to be popped from the front
  ## of the queue. An optional `timeout` in seconds can be used to limit
  ## the time spent waiting for an element. If the wait times out, an
  ## exception is raised.

proc clear*(this: var Queue): int  {.importcpp: "clear".}
  ## Empties the queue and returns the number of elements that were present
  ## at the start of the function. No threads are notified about this event
  ## as it is assumed to be used to drain the queue during shutdown of a
  ## `DataLoader`.

{.pop.} # header: "data/detail/queue.h
