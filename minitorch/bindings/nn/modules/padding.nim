{.push header: "nn/modules/padding.h".}


# Constructors and methods
proc reset*(this: var ReflectionPadImpl)  {.importcpp: "reset".}

proc forward*(this: var ReflectionPadImpl): int  {.importcpp: "forward".}

proc pretty_print*(this: ReflectionPadImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `ReflectionPad{1,2}d` module into the given
  ## `stream`.

proc reset*(this: var ReplicationPadImpl)  {.importcpp: "reset".}

proc forward*(this: var ReplicationPadImpl): int  {.importcpp: "forward".}

proc pretty_print*(this: ReplicationPadImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `ReplicationPad{1,2}d` module into the given
  ## `stream`.

proc reset*(this: var ZeroPad2dImpl)  {.importcpp: "reset".}

proc pretty_print*(this: ZeroPad2dImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `ZeroPad2d` module into the given `stream`.

proc forward*(this: var ZeroPad2dImpl): int  {.importcpp: "forward".}

proc reset*(this: var ConstantPadImpl)  {.importcpp: "reset".}

proc forward*(this: var ConstantPadImpl): int  {.importcpp: "forward".}

proc pretty_print*(this: ConstantPadImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `ConstantPad{1,2}d` module into the given `stream`.

{.pop.} # header: "nn/modules/padding.h
