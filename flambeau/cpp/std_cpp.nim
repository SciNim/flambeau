# Flambeau
# Copyright (c) 2020 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/macros

# ############################################################
#
#                   C++ standard types wrapper
#
# ############################################################

# std::string
# -----------------------------------------------------------------------

{.push header: "<string>".}

type
  CppString* {.importcpp: "std::string", bycopy.} = object

func len*(s: CppString): int {.importcpp: "#.length()".}
  ## Returns the length of a C++ std::string
func data*(s: CppString): ptr char {.importcpp: "const_cast<char*>(#.data())".}
  ## Returns a pointer to the raw data of a C++ std::string
func cstring*(s: CppString): cstring {.importcpp: "#.c_str()"}

{.pop.}

# Interop
# ------------------------------
func `$`*(s: CppString): string =
  result = newString(s.len)
  copyMem(result[0].addr, s.data, s.len)


# std::shared_ptr<T>
# -----------------------------------------------------------------------

{.push header: "<memory>".}

type
  CppSharedPtr*[T]{.importcpp: "std::shared_ptr", bycopy.} = object

func make_shared*(T: typedesc): CppSharedPtr[T] {.importcpp: "std::make_shared<'*0>()".}

{.pop.}

# std::unique_ptr<T>
# -----------------------------------------------------------------------

{.push header: "<memory>".}

type
  CppUniquePtr*[T]{.importcpp: "std::unique_ptr", header: "<memory>", bycopy.} = object

# func `=copy`*[T](dst: var CppUniquePtr[T], src: CppUniquePtr[T]) {.error: "A unique ptr cannot be copied".}
# func `=destroy`*[T](dst: var CppUniquePtr[T]){.importcpp: "#.~'*1()".}
# func `=sink`*[T](dst: var CppUniquePtr[T], src: CppUniquePtr[T]){.importcpp: "# = std::move(#)".}
func make_unique*(T: typedesc): CppUniquePtr[T] {.importcpp: "std::make_unique<'*0>()".}

{.pop.}

# Seamless pointer access
# -----------------------------------------------------------------------
{.experimental: "dotOperators".}

# This returns var T but with strictFunc it shouldn't
func deref*[T](p: CppUniquePtr[T] or CppSharedPtr[T]): var T {.noInit, importcpp: "(* #)", header: "<memory>".}

macro `.()`*[T](p: CppUniquePtr[T] or CppSharedPtr[T], fieldOrFunc: untyped, args: varargs[untyped]): untyped =
  result = nnkCall.newTree(
    nnkDotExpr.newTree(
      newCall(bindSym"deref", p),
      fieldOrFunc
    )
  )
  copyChildrenTo(args, result)

# std::vector<T>
# -----------------------------------------------------------------------

{.push header: "<memory>".}

type
  CppVector*[T]{.importcpp"std::vector", header: "<vector>", bycopy.} = object

proc init*(V: type CppVector): V {.importcpp: "std::vector<'*0>()", header: "<vector>", constructor.}
proc init*(V: type CppVector, size: int): V {.importcpp: "std::vector<'*0>(#)", header: "<vector>", constructor.}
proc len*(v: CppVector): int {.importcpp: "#.size()", header: "<vector>".}
proc add*[T](v: var CppVector[T], elem: T){.importcpp: "#.push_back(#)", header: "<vector>".}
proc `[]`*[T](v: CppVector[T], idx: int): T{.importcpp: "#[#]", header: "<vector>".}
proc `[]`*[T](v: var CppVector[T], idx: int): var T{.importcpp: "#[#]", header: "<vector>".}

{.pop.}

# std::tuple
# -----------------------------------------------------------------------
#
# We can either use objects or HList (Heterogenous Lists) or variadic templates to represent C++ tuples.
# We use objects for simplicity but in that case we need to create one type per size used in libtorch

# _batch_norm_impl_index, _thnn_fused_lstm_cell_backward need an arity of 5

{.push header: "<tuple>".}
type
  CppTuple2* [T0, T1] {.importcpp: "std::tuple".}= object
  CppTuple3* [T0, T1, T2] {.importcpp: "std::tuple".} = object
  CppTuple4* [T0, T1, T2, T3] {.importcpp: "std::tuple".} = object
  CppTuple5* [T0, T1, T2, T3, T4] {.importcpp: "std::tuple".} = object

  CppTuple = CppTuple2|CppTuple3|CppTuple4|CppTuple5

func tupGet(index: int, tup: CppTuple, outT: type): outT {.importcpp: "std::get<#>(#)".}
  ## C++ get from tuple.
  ## We have to use this unnatural argument order at low-level
  ## and add an outType parameter for out type inference

macro typeGet(Tup: typed, elem: static int): untyped =
  ## Return type inference for tuple extraction
  let Ti = ident("T" & $elem)
  result = nnkDotExpr.newTree(Tup, Ti)

template get*(tup: CppTuple, index: static int): auto =
  ## Extract a value from a C++ tuple
  # Note: it's important to use template here, we don't want
  # an extra proc even inline as std::get(std::tuple) is probably
  # special-cased to not copy unnecessarily or trigger std::shared_ptr refcount
  bind tupGet, typeGet
  tupGet(index, tup, typeGet(tup.typeof, index))

{.pop.}
