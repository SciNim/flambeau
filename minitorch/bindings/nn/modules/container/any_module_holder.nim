{.push header: "nn/modules/container/any_module_holder.h".}


# Constructors and methods
proc constructor_AnyModuleHolder<ModuleType, ArgumentTypes...>*(module_: var int &): AnyModuleHolder {.constructor,importcpp: "AnyModuleHolder<ModuleType, ArgumentTypes...>(@)".}
  ## Constructs the `AnyModuleHolder` from a concrete module.

proc forward*(this: var AnyModulePlaceholder): int  {.importcpp: "forward".}
  ## The "erased" `forward()` method.

proc ptr*(this: var AnyModulePlaceholder): int  {.importcpp: "ptr".}
  ## Returns std::shared_ptr<Module> pointing to the erased module.

proc copy*(this: AnyModulePlaceholder): int  {.importcpp: "copy".}
  ## Returns a `AnyModulePlaceholder` with a shallow copy of this
  ## `AnyModule`.

proc clone_module*(this: AnyModulePlaceholder): int  {.importcpp: "clone_module".}
  ## Returns a `AnyModulePlaceholder` with a deep copy of this `AnyModule`.

proc forward*(this: var AnyModuleHolder): int  {.importcpp: "forward".}
  ## Calls `forward()` on the underlying module, casting each `AnyValue` in
  ## the argument vector to a concrete value.

proc ptr*(this: var AnyModuleHolder): int  {.importcpp: "ptr".}

proc copy*(this: AnyModuleHolder): int  {.importcpp: "copy".}

proc clone_module*(this: AnyModuleHolder): int  {.importcpp: "clone_module".}

{.pop.} # header: "nn/modules/container/any_module_holder.h
