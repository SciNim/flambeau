{.push header: "expanding_array.h".}


# Constructors and methods
proc constructor_ExpandingArray<D, T>*(list: std::initializer_list<T>): ExpandingArray {.constructor,importcpp: "ExpandingArray<D, T>(@)".}
  ## Constructs an `ExpandingArray` from an `initializer_list`. The extent
  ## of the length is checked against the `ExpandingArray`'s extent
  ## parameter `D` at runtime.

proc constructor_ExpandingArray<D, T>*(vec: cint): ExpandingArray {.constructor,importcpp: "ExpandingArray<D, T>(@)".}
  ## Constructs an `ExpandingArray` from an `std::vector`. The extent of
  ## the length is checked against the `ExpandingArray`'s extent parameter
  ## `D` at runtime.

proc constructor_ExpandingArray<D, T>*(values: at::ArrayRef<T>): ExpandingArray {.constructor,importcpp: "ExpandingArray<D, T>(@)".}
  ## Constructs an `ExpandingArray` from an `at::ArrayRef`. The extent of
  ## the length is checked against the `ExpandingArray`'s extent parameter
  ## `D` at runtime.

proc constructor_ExpandingArray<D, T>*(single_size: T): ExpandingArray {.constructor,importcpp: "ExpandingArray<D, T>(@)".}
  ## Constructs an `ExpandingArray` from a single value, which is repeated
  ## `D` times (where `D` is the extent parameter of the `ExpandingArray`).

proc constructor_ExpandingArray<D, T>*(values: std::array<T, D>): ExpandingArray {.constructor,importcpp: "ExpandingArray<D, T>(@)".}
  ## Constructs an `ExpandingArray` from a correctly sized `std::array`.

proc constructor_ExpandingArrayWithOptionalElem<D, T>*(list: std::initializer_list<T>): ExpandingArrayWithOptionalElem {.constructor,importcpp: "ExpandingArrayWithOptionalElem<D, T>(@)".}
  ## Constructs an `ExpandingArrayWithOptionalElem` from an
  ## `initializer_list` of the underlying type `T`. The extent of the
  ## length is checked against the `ExpandingArrayWithOptionalElem`'s
  ## extent parameter `D` at runtime.

proc constructor_ExpandingArrayWithOptionalElem<D, T>*(vec: cint): ExpandingArrayWithOptionalElem {.constructor,importcpp: "ExpandingArrayWithOptionalElem<D, T>(@)".}
  ## Constructs an `ExpandingArrayWithOptionalElem` from an `std::vector`
  ## of the underlying type `T`. The extent of the length is checked
  ## against the `ExpandingArrayWithOptionalElem`'s extent parameter `D` at
  ## runtime.

proc constructor_ExpandingArrayWithOptionalElem<D, T>*(values: at::ArrayRef<T>): ExpandingArrayWithOptionalElem {.constructor,importcpp: "ExpandingArrayWithOptionalElem<D, T>(@)".}
  ## Constructs an `ExpandingArrayWithOptionalElem` from an `at::ArrayRef`
  ## of the underlying type `T`. The extent of the length is checked
  ## against the `ExpandingArrayWithOptionalElem`'s extent parameter `D` at
  ## runtime.

proc constructor_ExpandingArrayWithOptionalElem<D, T>*(single_size: T): ExpandingArrayWithOptionalElem {.constructor,importcpp: "ExpandingArrayWithOptionalElem<D, T>(@)".}
  ## Constructs an `ExpandingArrayWithOptionalElem` from a single value of
  ## the underlying type `T`, which is repeated `D` times (where `D` is the
  ## extent parameter of the `ExpandingArrayWithOptionalElem`).

proc constructor_ExpandingArrayWithOptionalElem<D, T>*(values: std::array<T, D>): ExpandingArrayWithOptionalElem {.constructor,importcpp: "ExpandingArrayWithOptionalElem<D, T>(@)".}
  ## Constructs an `ExpandingArrayWithOptionalElem` from a correctly sized
  ## `std::array` of the underlying type `T`.

proc `*`*(this: var ExpandingArray): std::array<T, D>  {.importcpp: "`*`".}
  ## Accesses the underlying `std::array`.

proc `*`*(this: ExpandingArray): std::array<T, D>  {.importcpp: "`*`".}
  ## Accesses the underlying `std::array`.

proc `->`*(this: var ExpandingArray): std::array<T, D> *  {.importcpp: "`->`".}
  ## Accesses the underlying `std::array`.

proc `->`*(this: ExpandingArray): std::array<T, D> *  {.importcpp: "`->`".}
  ## Accesses the underlying `std::array`.

proc size*(this: ExpandingArray): int  {.importcpp: "size".}
  ## Returns the extent of the `ExpandingArray`.

{.pop.} # header: "expanding_array.h
