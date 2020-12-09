{.push header: "data/transforms/base.h".}


# Constructors and methods
proc apply_batch*(this: var BatchTransform, input_batch: InputBatch): OutputBatch  {.importcpp: "apply_batch".}
  ## Applies the transformation to the given `input_batch`.

proc apply*(this: var Transform, input: torch::data::transforms::Transform::InputType): torch::data::transforms::Transform::OutputType  {.importcpp: "apply".}
  ## Applies the transformation to the given `input`.

proc apply_batch*(this: var Transform): int  {.importcpp: "apply_batch".}
  ## Applies the `transformation` over the entire `input_batch`.

{.pop.} # header: "data/transforms/base.h
