{.push header: "nn/cloneable.h".}


# Constructors and methods
proc reset*(this: var Cloneable)  {.importcpp: "reset".}
  ## `reset()` must perform initialization of all members with reference
  ## semantics, most importantly parameters, buffers and submodules.

proc clone*(this: Cloneable): int  {.importcpp: "clone".}
  ## Performs a recursive "deep copy" of the `Module`, such that all
  ## parameters and submodules in the cloned module are different from
  ## those in the original module.

proc clone_*(this: var Cloneable, other: cint, device: cint)  {.importcpp: "clone_".}

{.pop.} # header: "nn/cloneable.h
