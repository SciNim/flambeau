{.push header: "nn/modules/container/any.h".}


# Constructors and methods
proc constructor_AnyModule*(): AnyModule {.constructor,importcpp: "AnyModule".}
  ## A default-constructed `AnyModule` is in an empty state.

proc constructor_AnyModule*(var torch::nn::AnyModule &): AnyModule {.constructor,importcpp: "AnyModule(@)".}
  ## Move construction and assignment is allowed, and follows the default
  ## behavior of move for `std::unique_ptr`.

proc constructor_AnyModule*(other: torch::nn::AnyModule): AnyModule {.constructor,importcpp: "AnyModule(@)".}
  ## Creates a shallow copy of an `AnyModule`.

proc constructor_AnyModule*(other: torch::nn::AnyModule): AnyModule {.constructor,importcpp: "AnyModule(@)".}
  ## Creates a shallow copy of an `AnyModule`.

proc `=`*(this: var AnyModule, var torch::nn::AnyModule &): torch::nn::AnyModule  {.importcpp: "`=`".}

proc `=`*(this: var AnyModule, other: torch::nn::AnyModule): torch::nn::AnyModule  {.importcpp: "`=`".}

proc clone*(this: AnyModule, device: cint): torch::nn::AnyModule  {.importcpp: "clone".}
  ## Creates a deep copy of an `AnyModule` if it contains a module, else an
  ## empty `AnyModule` if it is empty.

proc ptr*(this: AnyModule): int  {.importcpp: "ptr".}
  ## Returns a `std::shared_ptr` whose dynamic type is that of the
  ## underlying module.

proc type_info*(this: AnyModule): std::type_info  {.importcpp: "type_info".}
  ## Returns the `type_info` object of the contained value.

proc is_empty*(this: AnyModule): bool  {.importcpp: "is_empty".}
  ## Returns true if the `AnyModule` does not contain a module.

proc `=`*(this: var AnyModule, other: torch::nn::AnyModule): torch::nn::AnyModule  {.importcpp: "`=`".}

proc clone*(this: AnyModule, device: cint): torch::nn::AnyModule  {.importcpp: "clone".}

proc ptr*(this: AnyModule): int  {.importcpp: "ptr".}

proc type_info*(this: AnyModule): std::type_info  {.importcpp: "type_info".}
  ## Returns the `type_info` object of the contained value.

proc is_empty*(this: AnyModule): bool  {.importcpp: "is_empty".}
  ## Returns true if the `AnyModule` does not contain a module.

{.pop.} # header: "nn/modules/container/any.h
