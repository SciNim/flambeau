{.push header: "nn/modules/dropout.h".}


# Constructors and methods
proc constructor__DropoutNd<Derived>*(p: cdouble): _DropoutNd {.constructor,importcpp: "_DropoutNd<Derived>(@)".}

proc constructor__DropoutNd<Derived>*(options_: cint): _DropoutNd {.constructor,importcpp: "_DropoutNd<Derived>(@)".}

proc reset*(this: var _DropoutNd)  {.importcpp: "reset".}

proc forward*(this: var DropoutImpl): int  {.importcpp: "forward".}

proc pretty_print*(this: DropoutImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `Dropout` module into the given `stream`.

proc forward*(this: var Dropout2dImpl): int  {.importcpp: "forward".}

proc pretty_print*(this: Dropout2dImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `Dropout2d` module into the given `stream`.

proc forward*(this: var Dropout3dImpl): int  {.importcpp: "forward".}

proc pretty_print*(this: Dropout3dImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `Dropout3d` module into the given `stream`.

proc forward*(this: var AlphaDropoutImpl): int  {.importcpp: "forward".}

proc pretty_print*(this: AlphaDropoutImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `AlphaDropout` module into the given `stream`.

proc forward*(this: var FeatureAlphaDropoutImpl): int  {.importcpp: "forward".}

proc pretty_print*(this: FeatureAlphaDropoutImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `FeatureAlphaDropout` module into the given
  ## `stream`.

{.pop.} # header: "nn/modules/dropout.h
