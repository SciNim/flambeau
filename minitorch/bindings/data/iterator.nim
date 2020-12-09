{.push header: "data/iterator.h".}


# Constructors and methods
proc constructor_ValidIterator<Batch>*(next_batch: cint): ValidIterator {.constructor,importcpp: "ValidIterator<Batch>(@)".}

proc constructor_Iterator<Batch>*(impl: cint): Iterator {.constructor,importcpp: "Iterator<Batch>(@)".}

proc next*(this: var IteratorImpl)  {.importcpp: "next".}

proc get*(this: var IteratorImpl): Batch  {.importcpp: "get".}

proc `==`*(this: IteratorImpl, other: IteratorImpl<Batch>): bool  {.importcpp: "`==`".}

proc `==`*(this: IteratorImpl, other: ValidIterator<Batch>): bool  {.importcpp: "`==`".}

proc `==`*(this: IteratorImpl, other: SentinelIterator<Batch>): bool  {.importcpp: "`==`".}

proc next*(this: var ValidIterator)  {.importcpp: "next".}
  ## Fetches the next batch.

proc get*(this: var ValidIterator): Batch  {.importcpp: "get".}
  ## Returns the current batch. The precondition for this operation to not
  ## throw an exception is that it has been compared to the
  ## `SentinelIterator` and did not compare equal.

proc `==`*(this: ValidIterator, other: IteratorImpl<Batch>): bool  {.importcpp: "`==`".}
  ## Does double dispatch.

proc `==`*(this: ValidIterator, SentinelIterator<Batch>): bool  {.importcpp: "`==`".}
  ## A `ValidIterator` is equal to the `SentinelIterator` iff. the
  ## `ValidIterator` has reached the end of the dataloader.

proc `==`*(this: ValidIterator, other: ValidIterator<Batch>): bool  {.importcpp: "`==`".}
  ## Returns true if the memory address of `other` equals that of `this`.

proc lazy_initialize*(this: ValidIterator)  {.importcpp: "lazy_initialize".}
  ## Gets the very first batch if it has not yet been fetched.

proc next*(this: var SentinelIterator)  {.importcpp: "next".}

proc get*(this: var SentinelIterator): Batch  {.importcpp: "get".}

proc `==`*(this: SentinelIterator, other: IteratorImpl<Batch>): bool  {.importcpp: "`==`".}
  ## Does double dispatch.

proc `==`*(this: SentinelIterator, other: ValidIterator<Batch>): bool  {.importcpp: "`==`".}
  ## Calls the comparison operator between `ValidIterator` and
  ## `SentinelIterator`.

proc `==`*(this: SentinelIterator, other: SentinelIterator<Batch>): bool  {.importcpp: "`==`".}
  ## Sentinel iterators always compare equal.

proc `++`*(this: var Iterator): Iterator<Batch>  {.importcpp: "`++`".}
  ## Increments the iterator. Only permitted for valid iterators (not past
  ## the end).

proc `*`*(this: var Iterator): Batch  {.importcpp: "`*`".}
  ## Returns the current batch. Only permitted for valid iterators (not
  ## past the end).

proc `->`*(this: var Iterator): Batch *  {.importcpp: "`->`".}
  ## Returns a pointer to the current batch. Only permitted for valid
  ## iterators (not past the end).

proc `==`*(this: Iterator, other: Iterator<Batch>): bool  {.importcpp: "`==`".}
  ## Compares two iterators for equality.

proc `!=`*(this: Iterator, other: Iterator<Batch>): bool  {.importcpp: "`!=`".}
  ## Compares two iterators for inequality.

{.pop.} # header: "data/iterator.h
