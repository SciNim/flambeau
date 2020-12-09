{.push header: "nn/modules/container/any_value.h".}


# Constructors and methods
proc constructor_AnyValue*(var torch::nn::AnyValue &): AnyValue {.constructor,importcpp: "AnyValue(@)".}
  ## Move construction and assignment is allowed, and follows the default
  ## behavior of move for `std::unique_ptr`.

proc constructor_AnyValue*(other: torch::nn::AnyValue): AnyValue {.constructor,importcpp: "AnyValue(@)".}
  ## Copy construction and assignment is allowed.

proc constructor_Placeholder*(type_info_: std::type_info): Placeholder {.constructor,importcpp: "Placeholder(@)".}

proc `=`*(this: var AnyValue, var torch::nn::AnyValue &): torch::nn::AnyValue  {.importcpp: "`=`".}

proc `=`*(this: var AnyValue, other: torch::nn::AnyValue): torch::nn::AnyValue  {.importcpp: "`=`".}

proc type_info*(this: AnyValue): std::type_info  {.importcpp: "type_info".}
  ## Returns the `type_info` object of the contained value.

proc clone*(this: Placeholder): int  {.importcpp: "clone".}

proc clone*(this: Holder): int  {.importcpp: "clone".}

{.pop.} # header: "nn/modules/container/any_value.h
