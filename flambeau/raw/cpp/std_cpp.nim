# Flambeau
# Copyright (c) 2020 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/macros
import ../bindings/c10
import cppstl
export cppstl

# ############################################################
#
#                   C++ standard types wrapper
#
# ############################################################


# std::string
# -----------------------------------------------------------------------
# See cppstl

# std::shared_ptr<T>
# -----------------------------------------------------------------------
# See cppstl

# std::unique_ptr<T>
# -----------------------------------------------------------------------
# See cppstl

# std::vector<T>
# -----------------------------------------------------------------------
# See cppstl

# std::tuple
# -----------------------------------------------------------------------
#
# We can either use objects or HList (Heterogenous Lists) or variadic templates to represent C++ tuples.
# We use objects for simplicity but in that case we need to create one type per size used in libtorch

# _batch_norm_impl_index, _thnn_fused_lstm_cell_backward need an arity of 5

{.push header: "<tuple>".}
type
  CppTuple2*[T0, T1] {.importcpp: "std::tuple".} = object
  CppTuple3*[T0, T1, T2] {.importcpp: "std::tuple".} = object
  CppTuple4*[T0, T1, T2, T3] {.importcpp: "std::tuple".} = object
  CppTuple5*[T0, T1, T2, T3, T4] {.importcpp: "std::tuple".} = object

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

converter toCppComplex*[T: SomeFloat](c: C10_Complex[T]): CppComplex[T] {.inline.} =
  result = initCppComplex(c.real(), c.imag())

# converter toC10_Complex*[T: SomeFloat](c: CppComplex[T]): C10_Complex[T] {.inline.} =
#   result = initC10_Complex(c.real(), c.imag())

proc `$`*[T: SomeFloat](z: C10_Complex[T]): string =
  result.add "("
  result.add $(z.real())
  result.add ", "
  result.add $(z.imag())
  result.add ")"
