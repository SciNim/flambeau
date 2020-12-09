{.push header: "serialize/output-archive.h".}


# Constructors and methods
proc constructor_OutputArchive*(cu: std::shared_ptr<jit::CompilationUnit>): OutputArchive {.constructor,importcpp: "OutputArchive(@)".}

proc constructor_OutputArchive*(): OutputArchive {.constructor,importcpp: "OutputArchive".}

proc constructor_OutputArchive*(var torch::serialize::OutputArchive &): OutputArchive {.constructor,importcpp: "OutputArchive(@)".}

proc constructor_OutputArchive*(var torch::serialize::OutputArchive): OutputArchive {.constructor,importcpp: "OutputArchive(@)".}

proc `=`*(this: var OutputArchive, var torch::serialize::OutputArchive &): torch::serialize::OutputArchive  {.importcpp: "`=`".}

proc `=`*(this: var OutputArchive, var torch::serialize::OutputArchive): torch::serialize::OutputArchive  {.importcpp: "`=`".}

proc compilation_unit*(this: OutputArchive): std::shared_ptr<jit::CompilationUnit>  {.importcpp: "compilation_unit".}

proc write*(this: var OutputArchive, key: std::string, ivalue: c10::IValue)  {.importcpp: "write".}
  ## Writes an `IValue` to the `OutputArchive`.

proc write*(this: var OutputArchive, key: std::string, tensor: at::Tensor, is_buffer: bool)  {.importcpp: "write".}
  ## Writes a `(key, tensor)` pair to the `OutputArchive`, and marks it as
  ## being or not being a buffer (non-differentiable tensor).

proc write*(this: var OutputArchive, key: std::string, nested_archive: var torch::serialize::OutputArchive)  {.importcpp: "write".}
  ## Writes a nested `OutputArchive` under the given `key` to this
  ## `OutputArchive`.

proc save_to*(this: var OutputArchive, filename: std::string)  {.importcpp: "save_to".}
  ## Saves the `OutputArchive` into a serialized representation in a file
  ## at `filename`.

proc save_to*(this: var OutputArchive, stream: var std::ostream)  {.importcpp: "save_to".}
  ## Saves the `OutputArchive` into a serialized representation into the
  ## given `stream`.

proc save_to*(this: var OutputArchive, func: cint)  {.importcpp: "save_to".}
  ## Saves the `OutputArchive` into a serialized representation using the
  ## given writer function.

{.pop.} # header: "serialize/output-archive.h
