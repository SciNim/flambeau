{.push header: "nn/pimpl.h".}


# Constructors and methods
proc constructor_ModuleHolder<Contained>*(): ModuleHolder {.constructor,importcpp: "ModuleHolder<Contained>".}
  ## Default constructs the contained module if if has a default
  ## constructor, else produces a static error.

proc constructor_ModuleHolder<Contained>*(std::nullptr_t): ModuleHolder {.constructor,importcpp: "ModuleHolder<Contained>(@)".}
  ## Constructs the `ModuleHolder` with an empty contained value. Access to
  ## the underlying module is not permitted and will throw an exception,
  ## until a value is assigned.

proc constructor_ModuleHolder<Contained>*(module: std::shared_ptr<Contained>): ModuleHolder {.constructor,importcpp: "ModuleHolder<Contained>(@)".}
  ## Constructs the `ModuleHolder` from a pointer to the contained type.
  ## Example: `Linear(std::make_shared<LinearImpl>(...))`.

proc `->`*(this: var ModuleHolder): Contained *  {.importcpp: "`->`".}
  ## Forwards to the contained module.

proc `->`*(this: ModuleHolder): Contained *  {.importcpp: "`->`".}
  ## Forwards to the contained module.

proc `*`*(this: var ModuleHolder): Contained  {.importcpp: "`*`".}
  ## Returns a reference to the contained module.

proc `*`*(this: ModuleHolder): Contained  {.importcpp: "`*`".}
  ## Returns a const reference to the contained module.

proc ptr*(this: ModuleHolder): std::shared_ptr<Contained>  {.importcpp: "ptr".}
  ## Returns a shared pointer to the underlying module.

proc get*(this: var ModuleHolder): Contained *  {.importcpp: "get".}
  ## Returns a pointer to the underlying module.

proc get*(this: ModuleHolder): Contained *  {.importcpp: "get".}
  ## Returns a const pointer to the underlying module.

proc is_empty*(this: ModuleHolder): bool  {.importcpp: "is_empty".}
  ## Returns true if the `ModuleHolder` does not contain a module.

{.pop.} # header: "nn/pimpl.h
