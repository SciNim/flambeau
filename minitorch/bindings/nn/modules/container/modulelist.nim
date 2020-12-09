{.push header: "nn/modules/container/modulelist.h".}


# Constructors and methods
proc constructor_ModuleListImpl*(): ModuleListImpl {.constructor,importcpp: "ModuleListImpl".}

proc clone*(this: ModuleListImpl): int  {.importcpp: "clone".}
  ## Special cloning function for `ModuleList` because it does not use
  ## `reset()`.

proc reset*(this: var ModuleListImpl)  {.importcpp: "reset".}
  ## `reset()` is empty for `ModuleList`, since it does not have parameters
  ## of its own.

proc pretty_print*(this: ModuleListImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `ModuleList` module into the given `stream`.

proc push_back*(this: var ModuleListImpl, module: cint)  {.importcpp: "push_back".}

proc begin*(this: var ModuleListImpl): int  {.importcpp: "begin".}
  ## Returns an iterator to the start of the `ModuleList`.

proc begin*(this: ModuleListImpl): int  {.importcpp: "begin".}
  ## Returns a const iterator to the start of the `ModuleList`.

proc end*(this: var ModuleListImpl): int  {.importcpp: "end".}
  ## Returns an iterator to the end of the `ModuleList`.

proc end*(this: ModuleListImpl): int  {.importcpp: "end".}
  ## Returns a const iterator to the end of the `ModuleList`.

proc ptr*(this: ModuleListImpl): int  {.importcpp: "ptr".}
  ## Attempts to return a `std::shared_ptr` whose dynamic type is that of
  ## the underlying module at the given index. Throws an exception if the
  ## index is out of bounds.

proc `[]`*(this: ModuleListImpl): int  {.importcpp: "`[]`".}
  ## Like `ptr(index)`.

proc size*(this: ModuleListImpl): int  {.importcpp: "size".}
  ## The current size of the `ModuleList` container.

proc is_empty*(this: ModuleListImpl): bool  {.importcpp: "is_empty".}
  ## True if there are no modules in the `ModuleList`.

proc insert*(this: var ModuleListImpl, index: cint, module: cint)  {.importcpp: "insert".}

proc push_back_var*(this: var ModuleListImpl)  {.importcpp: "push_back_var".}
  ## The base case, when the list of modules is empty.

{.pop.} # header: "nn/modules/container/modulelist.h
