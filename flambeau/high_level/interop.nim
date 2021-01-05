# Flambeau
# Copyright (c) 2020 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/complex,
  ../raw_bindings/tensors

# #######################################################################
#
#               Interop between Torch and Nim
#
# #######################################################################

# ArrayRefs
# -----------------------------------------------------
# libtorch/include/c10/util/ArrayRef.h

template asNimView*[T](ar: ArrayRef[T]): openarray[T] =
  toOpenArray(ar.data.unsafeAddr, 0, ar.size.int - 1)

template asTorchView*[T](oa: openarray[T]): ArrayRef[T] =
  ArrayRef[T].init(oa[0].unsafeAddr, oa.len)

# Type map
# -----------------------------------------------------

func toScalarKind(T: typedesc): static ScalarKind =
  ## Maps a Nim type to Torch scalar kind
  when T is uint8|byte:
    kUint8
  elif T is int8:
    kInt8
  elif T is int16:
    kInt16
  elif T is int32 or (T is int and sizeof(int) == sizeof(int32)):
    kInt32
  elif T is int64 or (T is int and sizeof(int) == sizeof(int64)):
    kInt64
  elif T is float32:
    kFloat32
  elif T is float64:
    kFloat64
  elif T is Complex[float32]:
    kComplexF32
  elif T is Complex[float64]:
    kComplexF64
  elif T is bool:
    kBool
  else:
    {.error: "Unsupported type in libtorch: " & $T.}

# Tensor <-> Nim sequences
# -----------------------------------------------------

func toTensor*[T](oa: openarray[T]): Tensor =
  ## Convert an openarray to a Tensor
  ## Input:
  ##      - An array or a seq (can be nested)
  ## Result:
  ##      - A Tensor of the same shape
  let shape = [oa.len.int64]
  return from_blob(
    oa[0].unsafeAddr,
    asTorchView(shape),
    toScalarKind(T)
  )
