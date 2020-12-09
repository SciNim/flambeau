{.push header: "nn/modules/_functions.h".}


# Constructors and methods
proc forward*(this: var CrossMapLRN2d, ctx: torch::autograd::AutogradContext *, input: torch::autograd::Variable, options: cint): torch::autograd::Variable  {.importcpp: "forward".}

proc backward*(this: var CrossMapLRN2d): int  {.importcpp: "backward".}

{.pop.} # header: "nn/modules/_functions.h
