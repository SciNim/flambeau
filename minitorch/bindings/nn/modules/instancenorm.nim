{.push header: "nn/modules/instancenorm.h".}


# Constructors and methods
proc forward*(this: var InstanceNormImpl): int  {.importcpp: "forward".}

proc pretty_print*(this: InstanceNormImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `InstanceNorm{1,2,3}d` module into the given
  ## `stream`.

{.pop.} # header: "nn/modules/instancenorm.h
