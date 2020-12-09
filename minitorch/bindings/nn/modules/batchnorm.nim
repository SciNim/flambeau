{.push header: "nn/modules/batchnorm.h".}


# Constructors and methods
proc constructor_NormImplBase<D, Derived, DerivedOptions>*(options_: DerivedOptions): NormImplBase {.constructor,importcpp: "NormImplBase<D, Derived, DerivedOptions>(@)".}

proc _check_input_dim*(this: var NormImplBase, input: cint)  {.importcpp: "_check_input_dim".}

proc reset*(this: var NormImplBase)  {.importcpp: "reset".}

proc reset_running_stats*(this: var NormImplBase)  {.importcpp: "reset_running_stats".}

proc reset_parameters*(this: var NormImplBase)  {.importcpp: "reset_parameters".}

proc forward*(this: var BatchNormImplBase): int  {.importcpp: "forward".}

proc pretty_print*(this: BatchNormImplBase, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `BatchNorm{1,2,3}d` module into the given `stream`.

{.pop.} # header: "nn/modules/batchnorm.h
