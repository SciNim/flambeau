{.push header: "nn/module.h".}


# Constructors and methods
proc constructor_Module*(name: std::string): Module {.constructor,importcpp: "Module(@)".}
  ## Tells the base `Module` about the name of the submodule.

proc constructor_Module*(): Module {.constructor,importcpp: "Module".}
  ## Constructs the module without immediate knowledge of the submodule's
  ## name. The name of the submodule is inferred via RTTI (if possible) the
  ## first time `.name()` is invoked.

proc name*(this: Module): std::string  {.importcpp: "name".}
  ## Returns the name of the `Module`.

proc clone*(this: Module, device: cint): std::shared_ptr<Module>  {.importcpp: "clone".}
  ## Performs a recursive deep copy of the module and all its registered
  ## parameters, buffers and submodules.

proc apply*(this: var Module, function: torch::nn::Module::ModuleApplyFunction)  {.importcpp: "apply".}
  ## Applies the `function` to the `Module` and recursively to every
  ## submodule. The function must accept a `Module&`.

proc apply*(this: Module, function: torch::nn::Module::ConstModuleApplyFunction)  {.importcpp: "apply".}
  ## Applies the `function` to the `Module` and recursively to every
  ## submodule. The function must accept a `const Module&`.

proc apply*(this: var Module, function: torch::nn::Module::NamedModuleApplyFunction, name_prefix: std::string)  {.importcpp: "apply".}
  ## Applies the `function` to the `Module` and recursively to every
  ## submodule. The function must accept a `const std::string&` for the key
  ## of the module, and a `Module&`. The key of the module itself is the
  ## empty string. If `name_prefix` is given, it is prepended to every key
  ## as `<name_prefix>.<key>` (and just `name_prefix` for the module
  ## itself).

proc apply*(this: Module, function: torch::nn::Module::ConstNamedModuleApplyFunction, name_prefix: std::string)  {.importcpp: "apply".}
  ## Applies the `function` to the `Module` and recursively to every
  ## submodule. The function must accept a `const std::string&` for the key
  ## of the module, and a `const Module&`. The key of the module itself is
  ## the empty string. If `name_prefix` is given, it is prepended to every
  ## key as `<name_prefix>.<key>` (and just `name_prefix` for the module
  ## itself).

proc apply*(this: Module, function: torch::nn::Module::ModulePointerApplyFunction)  {.importcpp: "apply".}
  ## Applies the `function` to the `Module` and recursively to every
  ## submodule. The function must accept a `const
  ## std::shared_ptr<Module>&`.

proc apply*(this: Module, function: torch::nn::Module::NamedModulePointerApplyFunction, name_prefix: std::string)  {.importcpp: "apply".}
  ## Applies the `function` to the `Module` and recursively to every
  ## submodule. The function must accept a `const std::string&` for the key
  ## of the module, and a `const std::shared_ptr<Module>&`. The key of the
  ## module itself is the empty string. If `name_prefix` is given, it is
  ## prepended to every key as `<name_prefix>.<key>` (and just
  ## `name_prefix` for the module itself).

proc parameters*(this: Module): int  {.importcpp: "parameters".}
  ## Returns the parameters of this `Module` and if `recurse` is true, also
  ## recursively of every submodule.

proc named_parameters*(this: Module): int  {.importcpp: "named_parameters".}
  ## Returns an `OrderedDict` with the parameters of this `Module` along
  ## with their keys, and if `recurse` is true also recursively of every
  ## submodule.

proc buffers*(this: Module): int  {.importcpp: "buffers".}
  ## Returns the buffers of this `Module` and if `recurse` is true, also
  ## recursively of every submodule.

proc named_buffers*(this: Module): int  {.importcpp: "named_buffers".}
  ## Returns an `OrderedDict` with the buffers of this `Module` along with
  ## their keys, and if `recurse` is true also recursively of every
  ## submodule.

proc modules*(this: Module): int  {.importcpp: "modules".}
  ## Returns the submodules of this `Module` (the entire submodule
  ## hierarchy) and if `include_self` is true, also inserts a `shared_ptr`
  ## to this module in the first position.

proc named_modules*(this: Module): int  {.importcpp: "named_modules".}
  ## Returns an `OrderedDict` of the submodules of this `Module` (the
  ## entire submodule hierarchy) and their keys, and if `include_self` is
  ## true, also inserts a `shared_ptr` to this module in the first
  ## position. If `name_prefix` is given, it is prepended to every key as
  ## `<name_prefix>.<key>` (and just `name_prefix` for the module itself).

proc children*(this: Module): int  {.importcpp: "children".}
  ## Returns the direct submodules of this `Module`.

proc named_children*(this: Module): int  {.importcpp: "named_children".}
  ## Returns an `OrderedDict` of the direct submodules of this `Module` and
  ## their keys.

proc train*(this: var Module, on: bool)  {.importcpp: "train".}
  ## Enables "training" mode.

proc eval*(this: var Module)  {.importcpp: "eval".}
  ## Calls train(false) to enable "eval" mode. Do not override this method,
  ## override `train()` instead.

proc is_training*(this: Module): bool  {.importcpp: "is_training".}
  ## True if the module is in training mode.

proc to*(this: var Module, device: cint, dtype: cint, non_blocking: bool)  {.importcpp: "to".}
  ## Recursively casts all parameters to the given `dtype` and `device`.

proc to*(this: var Module, dtype: cint, non_blocking: bool)  {.importcpp: "to".}
  ## Recursively casts all parameters to the given dtype.

proc to*(this: var Module, device: cint, non_blocking: bool)  {.importcpp: "to".}
  ## Recursively moves all parameters to the given device.

proc zero_grad*(this: var Module)  {.importcpp: "zero_grad".}
  ## Recursively zeros out the `grad` value of each registered parameter.

proc save*(this: Module, archive: cint)  {.importcpp: "save".}
  ## Serializes the `Module` into the given `OutputArchive`.

proc load*(this: var Module, archive: cint)  {.importcpp: "load".}
  ## Deserializes the `Module` from the given `InputArchive`.

proc pretty_print*(this: Module, stream: var std::ostream)  {.importcpp: "pretty_print".}
  ## Streams a pretty representation of the `Module` into the given
  ## `stream`. By default, this representation will be the name of the
  ## module (taken from `name()`), followed by a recursive pretty print of
  ## all of the `Module`'s submodules.

proc is_serializable*(this: Module): bool  {.importcpp: "is_serializable".}
  ## Returns whether the `Module` is serializable.

proc register_parameter*(this: var Module): int  {.importcpp: "register_parameter".}
  ## Registers a parameter with this `Module`.

proc register_buffer*(this: var Module): int  {.importcpp: "register_buffer".}
  ## Registers a buffer with this `Module`.

proc unregister_module*(this: var Module, name: std::string)  {.importcpp: "unregister_module".}
  ## Unregisters a submodule from this `Module`. If there is no such module
  ## with `name` an exception is thrown.

proc _forward_has_default_args*(this: var Module): bool  {.importcpp: "_forward_has_default_args".}
  ## The following three functions allow a module with default arguments in
  ## its forward method to be used in a Sequential module. You should NEVER
  ## override these functions manually. Instead, you should use the
  ## `FORWARD_HAS_DEFAULT_ARGS` macro.

proc _forward_num_required_args*(this: var Module): unsigned int  {.importcpp: "_forward_num_required_args".}

proc _forward_populate_default_args*(this: var Module): int  {.importcpp: "_forward_populate_default_args".}

proc clone_*(this: var Module, other: var torch::nn::Module, device: cint)  {.importcpp: "clone_".}
  ## Used in the implementation of `Cloneable`.

proc pretty_print_recursive*(this: Module, stream: var std::ostream, indentation: std::string)  {.importcpp: "pretty_print_recursive".}
  ## Implements pretty printing the module hierarchy.

proc apply_to_submodules*(this: Module, function: torch::nn::Module::NamedModulePointerApplyFunction, name_prefix: std::string)  {.importcpp: "apply_to_submodules".}
  ## Applies the `function` to every submodule recursively, starting at
  ## this `Module`'s children (thus not including the module itself).

proc shared_from_this_checked*(this: Module): std::shared_ptr<Module>  {.importcpp: "shared_from_this_checked".}
  ## Returns a shared_ptr to `this` in a safe (checked) way.

{.pop.} # header: "nn/module.h
