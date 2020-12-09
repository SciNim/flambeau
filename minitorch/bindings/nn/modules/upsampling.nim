{.push header: "nn/modules/upsampling.h".}


# Constructors and methods
proc constructor_UpsampleImpl*(options_: cint): UpsampleImpl {.constructor,importcpp: "UpsampleImpl(@)".}

proc reset*(this: var UpsampleImpl)  {.importcpp: "reset".}

proc pretty_print*(this: UpsampleImpl, stream: var std::ostream)  {.importcpp: "pretty_print".}
  ## Pretty prints the `Upsample` module into the given `stream`.

proc forward*(this: var UpsampleImpl): int  {.importcpp: "forward".}

{.pop.} # header: "nn/modules/upsampling.h
