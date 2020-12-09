{.push header: "nn/modules/container/sequential.h".}


# Constructors and methods
proc constructor_SequentialImpl*(): SequentialImpl {.constructor,importcpp: "SequentialImpl".}

proc constructor_SequentialImpl*(ordered_dict: var int &): SequentialImpl {.constructor,importcpp: "SequentialImpl(@)".}
  ## Constructs the `Sequential` from an `OrderedDict` of named
  ## `AnyModule`s.

proc constructor_SequentialImpl*(named_modules: cint): SequentialImpl {.constructor,importcpp: "SequentialImpl(@)".}
  ## Constructs the `Sequential` from a braced-init-list of named
  ## `AnyModule`s. It enables the following use case: `Sequential
  ## sequential({{"m1", M(1)}, {"m2", M(2)}})`

proc constructor_Sequential*(): Sequential {.constructor,importcpp: "Sequential".}

proc constructor_Sequential*(named_modules: cint): Sequential {.constructor,importcpp: "Sequential(@)".}
  ## Constructs the `Sequential` from a braced-init-list of named
  ## `AnyModule`s. It enables the following use case: `Sequential
  ## sequential({{"m1", M(1)}, {"m2", M(2)}})`

proc clone*(this: SequentialImpl): int  {.importcpp: "clone".}
  ## Special cloning function for `Sequential` because it does not use
  ## `reset()`.

proc reset*(this: var SequentialImpl)  {.importcpp: "reset".}
  ## `reset()` is empty for `Sequential`, since it does not have parameters
  ## of its own.

proc pretty_print*(this: SequentialImpl, stream: var std::ostream)  {.importcpp: "pretty_print".}
  ## Pretty prints the `Sequential` module into the given `stream`.

proc push_back*(this: var SequentialImpl, any_module: cint)  {.importcpp: "push_back".}
  ## Adds a type-erased `AnyModule` to the `Sequential`.

proc push_back*(this: var SequentialImpl, name: std::string, any_module: cint)  {.importcpp: "push_back".}

proc begin*(this: var SequentialImpl): int  {.importcpp: "begin".}
  ## Returns an iterator to the start of the `Sequential`.

proc begin*(this: SequentialImpl): int  {.importcpp: "begin".}
  ## Returns a const iterator to the start of the `Sequential`.

proc end*(this: var SequentialImpl): int  {.importcpp: "end".}
  ## Returns an iterator to the end of the `Sequential`.

proc end*(this: SequentialImpl): int  {.importcpp: "end".}
  ## Returns a const iterator to the end of the `Sequential`.

proc ptr*(this: SequentialImpl): int  {.importcpp: "ptr".}
  ## Attempts to return a `std::shared_ptr` whose dynamic type is that of
  ## the underlying module at the given index. Throws an exception if the
  ## index is out of bounds.

proc `[]`*(this: SequentialImpl): int  {.importcpp: "`[]`".}
  ## Like `ptr(index)`.

proc size*(this: SequentialImpl): int  {.importcpp: "size".}
  ## The current size of the `Sequential` container.

proc is_empty*(this: SequentialImpl): bool  {.importcpp: "is_empty".}
  ## True if there are no modules in the `Sequential`.

proc push_back*(this: var SequentialImpl)  {.importcpp: "push_back".}
  ## The base case, when the list of modules is empty.

{.pop.} # header: "nn/modules/container/sequential.h
