{.push header: "data/example.h".}


# Constructors and methods
proc constructor_Example<Data, Target>*(): Example {.constructor,importcpp: "Example<Data, Target>".}

proc constructor_Example<Data, Target>*(data: Data, target: Target): Example {.constructor,importcpp: "Example<Data, Target>(@)".}

proc constructor_Example<type-parameter-0-0, void>*(): Example {.constructor,importcpp: "Example<type-parameter-0-0, void>".}

proc constructor_Example<type-parameter-0-0, void>*(data: Data): Example {.constructor,importcpp: "Example<type-parameter-0-0, void>(@)".}

{.pop.} # header: "data/example.h
