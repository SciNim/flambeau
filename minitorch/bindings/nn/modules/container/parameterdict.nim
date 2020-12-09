{.push header: "nn/modules/container/parameterdict.h".}


# Constructors and methods
proc constructor_ParameterDictImpl*(): ParameterDictImpl {.constructor,importcpp: "ParameterDictImpl".}

proc constructor_ParameterDictImpl*(params: cint): ParameterDictImpl {.constructor,importcpp: "ParameterDictImpl(@)".}

proc reset*(this: var ParameterDictImpl)  {.importcpp: "reset".}
  ## `reset()` is empty for `ParameterDict`, since it does not have
  ## parameters of its own.

proc pretty_print*(this: ParameterDictImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `ParameterDict` module into the given `stream`.

proc insert*(this: var ParameterDictImpl): int  {.importcpp: "insert".}
  ## Insert the parameter along with the key into ParameterDict The
  ## parameter is set to be require grad by default

proc pop*(this: var ParameterDictImpl): int  {.importcpp: "pop".}
  ## Remove key from the ParameterDict and return its value, throw
  ## exception if the key is not contained. Please check contains(key)
  ## before for a non-throwing access.

proc keys*(this: ParameterDictImpl): int  {.importcpp: "keys".}
  ## Return the keys in the dict

proc values*(this: ParameterDictImpl): int  {.importcpp: "values".}
  ## Return the Values in the dict

proc begin*(this: var ParameterDictImpl): int  {.importcpp: "begin".}
  ## Return an iterator to the start of ParameterDict

proc begin*(this: ParameterDictImpl): int  {.importcpp: "begin".}
  ## Return a const iterator to the start of ParameterDict

proc end*(this: var ParameterDictImpl): int  {.importcpp: "end".}
  ## Return an iterator to the end of ParameterDict

proc end*(this: ParameterDictImpl): int  {.importcpp: "end".}
  ## Return a const iterator to the end of ParameterDict

proc size*(this: ParameterDictImpl): int  {.importcpp: "size".}
  ## Return the number of items currently stored in the ParameterDict

proc empty*(this: ParameterDictImpl): bool  {.importcpp: "empty".}
  ## Return true if the ParameterDict is empty, otherwise return false

proc clear*(this: var ParameterDictImpl)  {.importcpp: "clear".}
  ## Remove all parameters in the ParameterDict

proc contains*(this: ParameterDictImpl, key: cint): bool  {.importcpp: "contains".}
  ## Check if the centain parameter with the key in the ParameterDict

proc get*(this: ParameterDictImpl): int  {.importcpp: "get".}
  ## Returns the value associated with the given `key`. Throws an exception
  ## if no such key is stored in the `ParameterDict`. Check contains(key)
  ## before for a non-throwing way of access

proc get*(this: var ParameterDictImpl): int  {.importcpp: "get".}
  ## Returns the value associated with the given `key`. Throws an exception
  ## if no such key is stored in the `ParameterDict`. Check contains(key)
  ## before for a non-throwing way of access

proc `[]`*(this: var ParameterDictImpl): int  {.importcpp: "`[]`".}
  ## Returns the value associated with the given `key`. Throws an exception
  ## if no such key is stored in the `ParameterDict`. Check contains(key)
  ## before for a non-throwing way of access

proc `[]`*(this: ParameterDictImpl): int  {.importcpp: "`[]`".}
  ## Returns the value associated with the given `key`. Throws an exception
  ## if no such key is stored in the `ParameterDict`. Check contains(key)
  ## before for a non-throwing way of access

{.pop.} # header: "nn/modules/container/parameterdict.h
