{.push header: "optim/optimizer.h".}


# Constructors and methods
proc constructor_OptimizerParamGroup*(param_group: torch::optim::OptimizerParamGroup): OptimizerParamGroup {.constructor,importcpp: "OptimizerParamGroup(@)".}

proc constructor_OptimizerParamGroup*(params: cint): OptimizerParamGroup {.constructor,importcpp: "OptimizerParamGroup(@)".}

proc constructor_OptimizerParamGroup*(params: cint, options: cint): OptimizerParamGroup {.constructor,importcpp: "OptimizerParamGroup(@)".}

proc constructor_Optimizer*(optimizer: torch::optim::Optimizer): Optimizer {.constructor,importcpp: "Optimizer(@)".}

proc constructor_Optimizer*(optimizer: var torch::optim::Optimizer &): Optimizer {.constructor,importcpp: "Optimizer(@)".}

proc constructor_Optimizer*(param_groups: cint, defaults: cint): Optimizer {.constructor,importcpp: "Optimizer(@)".}

proc constructor_Optimizer*(parameters: cint, defaults: cint): Optimizer {.constructor,importcpp: "Optimizer(@)".}
  ## Constructs the `Optimizer` from a vector of parameters.

proc clone*(this: OptimizerParamState): int  {.importcpp: "clone".}

proc serialize*(this: var OptimizerParamState, archive: var torch::serialize::InputArchive)  {.importcpp: "serialize".}

proc serialize*(this: OptimizerParamState, archive: var torch::serialize::OutputArchive)  {.importcpp: "serialize".}

proc clone*(this: OptimizerCloneableParamState): int  {.importcpp: "clone".}

proc clone*(this: OptimizerOptions): int  {.importcpp: "clone".}

proc serialize*(this: var OptimizerOptions, archive: var torch::serialize::InputArchive)  {.importcpp: "serialize".}

proc serialize*(this: OptimizerOptions, archive: var torch::serialize::OutputArchive)  {.importcpp: "serialize".}

proc clone*(this: OptimizerCloneableOptions): int  {.importcpp: "clone".}

proc has_options*(this: OptimizerParamGroup): bool  {.importcpp: "has_options".}

proc options*(this: var OptimizerParamGroup): torch::optim::OptimizerOptions  {.importcpp: "options".}

proc options*(this: OptimizerParamGroup): torch::optim::OptimizerOptions  {.importcpp: "options".}

proc set_options*(this: var OptimizerParamGroup, options: cint)  {.importcpp: "set_options".}

proc params*(this: var OptimizerParamGroup): int  {.importcpp: "params".}

proc params*(this: OptimizerParamGroup): int  {.importcpp: "params".}

proc add_param_group*(this: var Optimizer, param_group: torch::optim::OptimizerParamGroup)  {.importcpp: "add_param_group".}
  ## Adds the given param_group to the optimizer's param_group list.

proc step*(this: var Optimizer, closure: torch::optim::Optimizer::LossClosure): at::Tensor  {.importcpp: "step".}
  ## A loss function closure, which is expected to return the loss value.

proc add_parameters*(this: var Optimizer, parameters: cint)  {.importcpp: "add_parameters".}
  ## Adds the given vector of parameters to the optimizer's parameter list.

proc zero_grad*(this: var Optimizer)  {.importcpp: "zero_grad".}
  ## Zeros out the gradients of all parameters.

proc parameters*(this: Optimizer): int  {.importcpp: "parameters".}
  ## Provides a const reference to the parameters in the first param_group
  ## this optimizer holds.

proc parameters*(this: var Optimizer): int  {.importcpp: "parameters".}
  ## Provides a reference to the parameters in the first param_group this
  ## optimizer holds.

proc size*(this: Optimizer): int  {.importcpp: "size".}
  ## Returns the number of parameters referenced by the optimizer.

proc defaults*(this: var Optimizer): torch::optim::OptimizerOptions  {.importcpp: "defaults".}

proc defaults*(this: Optimizer): torch::optim::OptimizerOptions  {.importcpp: "defaults".}

proc param_groups*(this: var Optimizer): int  {.importcpp: "param_groups".}
  ## Provides a reference to the param_groups this optimizer holds.

proc param_groups*(this: Optimizer): int  {.importcpp: "param_groups".}
  ## Provides a const reference to the param_groups this optimizer holds.

proc state*(this: var Optimizer): int  {.importcpp: "state".}
  ## Provides a reference to the state this optimizer holds

proc state*(this: Optimizer): int  {.importcpp: "state".}
  ## Provides a const reference to the state this optimizer holds

proc save*(this: Optimizer, archive: var serialize::OutputArchive)  {.importcpp: "save".}
  ## Serializes the optimizer state into the given `archive`.

proc load*(this: var Optimizer, archive: var serialize::InputArchive)  {.importcpp: "load".}
  ## Deserializes the optimizer state from the given `archive`.

{.pop.} # header: "optim/optimizer.h
