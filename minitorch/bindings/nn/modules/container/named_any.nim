{.push header: "nn/modules/container/named_any.h".}


# Constructors and methods
proc constructor_NamedAnyModule*(name: std::string, any_module: cint): NamedAnyModule {.constructor,importcpp: "NamedAnyModule(@)".}
  ## Creates a `NamedAnyModule` from a type-erased `AnyModule`.

proc name*(this: NamedAnyModule): std::string  {.importcpp: "name".}
  ## Returns a reference to the name.

proc module*(this: var NamedAnyModule): int  {.importcpp: "module".}
  ## Returns a reference to the module.

proc module*(this: NamedAnyModule): int  {.importcpp: "module".}
  ## Returns a const reference to the module.

{.pop.} # header: "nn/modules/container/named_any.h
