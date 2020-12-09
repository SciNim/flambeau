{.push header: "data/transforms/lambda.h".}


# Constructors and methods
proc constructor_BatchLambda<Input, Output>*(function: cint): BatchLambda {.constructor,importcpp: "BatchLambda<Input, Output>(@)".}
  ## Constructs the `BatchLambda` from the given `function` object.

proc constructor_Lambda<Input, Output>*(function: torch::data::transforms::Lambda::FunctionType): Lambda {.constructor,importcpp: "Lambda<Input, Output>(@)".}
  ## Constructs the `Lambda` from the given `function` object.

proc apply_batch*(this: var BatchLambda): int  {.importcpp: "apply_batch".}
  ## Applies the user-provided function object to the `input_batch`.

proc apply*(this: var Lambda): int  {.importcpp: "apply".}
  ## Applies the user-provided function object to the `input`.

{.pop.} # header: "data/transforms/lambda.h
