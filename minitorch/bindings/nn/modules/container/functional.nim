{.push header: "nn/modules/container/functional.h".}


# Constructors and methods
proc constructor_FunctionalImpl*(function: cint): FunctionalImpl {.constructor,importcpp: "FunctionalImpl(@)".}
  ## Constructs a `Functional` from a function object.

proc reset*(this: var FunctionalImpl)  {.importcpp: "reset".}

proc pretty_print*(this: FunctionalImpl, stream: var std::ostream)  {.importcpp: "pretty_print".}
  ## Pretty prints the `Functional` module into the given `stream`.

proc forward*(this: var FunctionalImpl): int  {.importcpp: "forward".}
  ## Forwards the `input` tensor to the underlying (bound) function object.

proc `()`*(this: var FunctionalImpl): int  {.importcpp: "`()`".}
  ## Calls forward(input).

proc is_serializable*(this: FunctionalImpl): bool  {.importcpp: "is_serializable".}

{.pop.} # header: "nn/modules/container/functional.h
