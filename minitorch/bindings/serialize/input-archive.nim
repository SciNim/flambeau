{.push header: "serialize/input-archive.h".}


# Constructors and methods
proc constructor_InputArchive*(): InputArchive {.constructor,importcpp: "InputArchive".}
  ## Default-constructs the `InputArchive`.

proc constructor_InputArchive*(var torch::serialize::InputArchive &): InputArchive {.constructor,importcpp: "InputArchive(@)".}

proc constructor_InputArchive*(var torch::serialize::InputArchive): InputArchive {.constructor,importcpp: "InputArchive(@)".}

proc `=`*(this: var InputArchive, var torch::serialize::InputArchive &): torch::serialize::InputArchive  {.importcpp: "`=`".}

proc `=`*(this: var InputArchive, var torch::serialize::InputArchive): torch::serialize::InputArchive  {.importcpp: "`=`".}

proc read*(this: var InputArchive, key: std::string, ivalue: var c10::IValue)  {.importcpp: "read".}
  ## Reads an `IValue` associated with a given `key`.

proc try_read*(this: var InputArchive, key: std::string, ivalue: var c10::IValue): bool  {.importcpp: "try_read".}
  ## Reads an `IValue` associated with a given `key`. If there is no
  ## `IValue` associated with the `key`, this returns false, otherwise it
  ## returns true.

proc try_read*(this: var InputArchive, key: std::string, tensor: var at::Tensor, is_buffer: bool): bool  {.importcpp: "try_read".}
  ## Reads a `tensor` associated with a given `key`. If there is no
  ## `tensor` associated with the `key`, this returns false, otherwise it
  ## returns true. If the tensor is expected to be a buffer (not
  ## differentiable), `is_buffer` must be `true`.

proc read*(this: var InputArchive, key: std::string, tensor: var at::Tensor, is_buffer: bool)  {.importcpp: "read".}
  ## Reads a `tensor` associated with a given `key`. If the tensor is
  ## expected to be a buffer (not differentiable), `is_buffer` must be
  ## `true`.

proc try_read*(this: var InputArchive, key: std::string, archive: var torch::serialize::InputArchive): bool  {.importcpp: "try_read".}
  ## Reads a `InputArchive` associated with a given `key`. If there is no
  ## `InputArchive` associated with the `key`, this returns false,
  ## otherwise it returns true.

proc read*(this: var InputArchive, key: std::string, archive: var torch::serialize::InputArchive)  {.importcpp: "read".}
  ## Reads an `InputArchive` associated with a given `key`. The archive can
  ## thereafter be used for further deserialization of the nested data.

proc load_from*(this: var InputArchive, filename: std::string, device: cint)  {.importcpp: "load_from".}
  ## Loads the `InputArchive` from a serialized representation stored in
  ## the file at `filename`. Storage are remapped using device option. If
  ## device is not specified, the module is loaded to the original device.

proc load_from*(this: var InputArchive, stream: var std::istream, device: cint)  {.importcpp: "load_from".}
  ## Loads the `InputArchive` from a serialized representation stored in
  ## the given `stream`. Storage are remapped using device option. If
  ## device is not specified, the module is loaded to the original device.

proc load_from*(this: var InputArchive, data: char *, size: cint, device: cint)  {.importcpp: "load_from".}

proc load_from*(this: var InputArchive, read_func: cint, size_func: cint, device: cint)  {.importcpp: "load_from".}

proc keys*(this: var InputArchive): int  {.importcpp: "keys".}

{.pop.} # header: "serialize/input-archive.h
