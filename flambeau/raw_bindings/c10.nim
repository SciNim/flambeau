# Flambeau
# Copyright (c) 2020 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/complex,
  ../libtorch,
  ../cpp/std_cpp

# c10 is a collection of utilities in PyTorch

# C++ interop
# -----------------------------------------------------------------------

{.push cdecl.}
{.push header: torchHeader.}

# ArrayRef
# -----------------------------------------------------------------------
#
# LibTorch is using "ArrayRef" through the codebase in particular
# for shapes and strides.
#
# It has the following definition in
# libtorch/include/c10/util/ArrayRef.h
#
# template <typename T>
# class ArrayRef final {
#  private:
#   /// The start of the array, in an external buffer.
#   const T* Data;
#
#   /// The number of elements.
#   size_type Length;
#
# It is noted that the class does not own the underlying data.
# We can model that in a zero-copy and safely borrow-checked way
# with "openarray[T]"

{.experimental: "views".} # TODO this is ignored

type
  ArrayRef*[T] {.importcpp: "c10::ArrayRef", bycopy.} = object
    # The field are private so we can't use them, but `lent` enforces borrow checking
    p: lent UncheckedArray[T]
    len: csize_t

  IntArrayRef* = ArrayRef[int64]

func data*[T](ar: ArrayRef[T]): ptr UncheckedArray[T] {.importcpp: "#.data()".}
func size*(ar: ArrayRef): csize_t {.importcpp: "#.size()".}

func init*[T](AR: type ArrayRef[T], p: ptr T, len: SomeInteger): ArrayRef[T] {.constructor, importcpp: "c10::ArrayRef<'*0>(@)".}
func init*[T](AR: type ArrayRef[T]): ArrayRef[T] {.constructor, varargs, importcpp: "c10::ArrayRef<'*0>({@})".}

# Optional
# -----------------------------------------------------------------------

type
  Optional*[T] {.bycopy, importcpp: "c10::optional".} = object
  Nullopt_t {.bycopy, importcpp: "c10::nullopt_t".} = object

func value*[T](o: Optional[T]): T {.importcpp: "#.value()".}
