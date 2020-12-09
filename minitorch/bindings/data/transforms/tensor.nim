{.push header: "data/transforms/tensor.h".}


# Constructors and methods
proc constructor_TensorLambda<Target>*(function: cint): TensorLambda {.constructor,importcpp: "TensorLambda<Target>(@)".}
  ## Creates a `TensorLambda` from the given `function`.

proc `()`*(this: var TensorTransform): int  {.importcpp: "`()`".}
  ## Transforms a single input tensor to an output tensor.

proc apply*(this: var TensorTransform): int  {.importcpp: "apply".}
  ## Implementation of `Transform::apply` that calls `operator()`.

proc `()`*(this: var TensorLambda): int  {.importcpp: "`()`".}
  ## Applies the user-provided functor to the input tensor.

proc stddev*(this: var Normalize, stddev: cint): Normalize<Target>  {.importcpp: "stddev".}
  ## Constructs a `Normalize` transform. The mean and standard deviation
  ## can be anything that is broadcastable over the input tensors (like
  ## single scalars).

{.pop.} # header: "data/transforms/tensor.h
