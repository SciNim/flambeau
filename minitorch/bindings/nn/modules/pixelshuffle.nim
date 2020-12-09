{.push header: "nn/modules/pixelshuffle.h".}


# Constructors and methods
proc constructor_PixelShuffleImpl*(options_: cint): PixelShuffleImpl {.constructor,importcpp: "PixelShuffleImpl(@)".}

proc pretty_print*(this: PixelShuffleImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `PixelShuffle` module into the given `stream`.

proc forward*(this: var PixelShuffleImpl): int  {.importcpp: "forward".}

proc reset*(this: var PixelShuffleImpl)  {.importcpp: "reset".}

{.pop.} # header: "nn/modules/pixelshuffle.h
