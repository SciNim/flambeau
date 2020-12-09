{.push header: "nn/modules/container/parameterlist.h".}


# Constructors and methods
proc constructor_ParameterListImpl*(): ParameterListImpl {.constructor,importcpp: "ParameterListImpl".}

proc reset*(this: var ParameterListImpl)  {.importcpp: "reset".}
  ## `reset()` is empty for `ParameterList`, since it does not have
  ## parameters of its own.

proc pretty_print*(this: ParameterListImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `ParameterList` module into the given `stream`.

proc append*(this: var ParameterListImpl, param: var int &)  {.importcpp: "append".}
  ## push the a given parameter at the end of the list

proc append*(this: var ParameterListImpl, param: cint)  {.importcpp: "append".}
  ## push the a given parameter at the end of the list

proc append*(this: var ParameterListImpl, pair: cint)  {.importcpp: "append".}
  ## push the a given parameter at the end of the list And the key of the
  ## pair will be discarded, only the value will be added into the
  ## `ParameterList`

proc begin*(this: var ParameterListImpl): int  {.importcpp: "begin".}
  ## Returns an iterator to the start of the ParameterList the iterator
  ## returned will be type of `OrderedDict<std::string,
  ## torch::Tensor>::Item`

proc begin*(this: ParameterListImpl): int  {.importcpp: "begin".}
  ## Returns a const iterator to the start of the ParameterList the
  ## iterator returned will be type of `OrderedDict<std::string,
  ## torch::Tensor>::Item`

proc end*(this: var ParameterListImpl): int  {.importcpp: "end".}
  ## Returns an iterator to the end of the ParameterList the iterator
  ## returned will be type of `OrderedDict<std::string,
  ## torch::Tensor>::Item`

proc end*(this: ParameterListImpl): int  {.importcpp: "end".}
  ## Returns a const iterator to the end of the ParameterList the iterator
  ## returned will be type of `OrderedDict<std::string,
  ## torch::Tensor>::Item`

proc at*(this: var ParameterListImpl): int  {.importcpp: "at".}
  ## Returns the value associated with the given `key`. Throws an exception
  ## if no such key is stored in the `ParameterList`. Check contains(key)
  ## before for a non-throwing way of access

{.pop.} # header: "nn/modules/container/parameterlist.h
