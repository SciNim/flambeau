# Flambeau
# Copyright (c) 2020 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/[complex, enumerate, typetraits, macros],
  ../raw_bindings/tensors,
  ./internal/dynamic_stack_arrays

# #######################################################################
#
#               Interop between Torch and Nim
#
# #######################################################################

type Metadata = DynamicStackArray[int64]

# ArrayRefs
# -----------------------------------------------------
# libtorch/include/c10/util/ArrayRef.h

{.experimental:"views".}

template asNimView*[T](ar: ArrayRef[T]): openarray[T] =
  toOpenArray(ar.data, 0, ar.size.int - 1)

template asTorchView*[T](oa: openarray[T]): ArrayRef[T] =
  ArrayRef[T].init(oa[0].unsafeAddr, oa.len)

template asTorchView(meta: Metadata): ArrayRef[int64] =
  ArrayRef[int64].init(meta.data[0].unsafeAddr, meta.len)

# Type map
# -----------------------------------------------------

func toScalarKind(T: typedesc[SomeTorchType]): static ScalarKind =
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

# Nim openarrays -> Torch Tensors
# -----------------------------------------------------

func getShape[T](s: openarray[T], parent_shape = Metadata()): Metadata =
  ## Get the shape of nested seqs/arrays
  ## Important ⚠: at each nesting level, only the length
  ##   of the first element is used for the shape.
  ##   Ensure before or after that seqs have the expected length
  ##   or that the total number of elements matches the product of the dimensions.

  result = parent_shape
  result.add(s.len)

  when (T is seq|array):
    result = getShape(s[0], result)

macro getBaseType(T: typedesc): untyped =
  # Get the base T of a seq[T] input
  result = T.getTypeInst()[1]
  while result.kind == nnkBracketExpr and (
          result[0].eqIdent"seq" or result[0].eqIdent"array"):
    # We can also have nnkBracketExpr(Complex, float32)
    if result[0].eqIdent"seq":
      result = result[1]
    else: # array
      result = result[2]

iterator flatIter*[T](s: openarray[T]): auto {.noSideEffect.}=
  ## Inline iterator on any-depth seq or array
  ## Returns values in order
  for item in s:
    when item is array|seq:
      for subitem in flatIter(item):
        yield subitem
    else:
      yield item

func toTensorView*[T: SomeTorchType](oa: openarray[T]): lent Tensor =
  ## Interpret an openarray as a CPU Tensor
  ## Important:
  ##   the buffer is shared.
  ##   There is no copy but modifications are shared
  ##   and the view cannot outlive its buffer.
  ##
  ## Input:
  ##      - An array or a seq (can be nested)
  ## Result:
  ##      - A view Tensor of the same shape
  return from_blob(
    oa[0].unsafeAddr,
    oa.len.int64,
    toScalarKind(T)
  )

func toTensor*[T: SomeTorchType](oa: openarray[T]): Tensor =
  ## Interpret an openarray as a CPU Tensor
  ##
  ## Input:
  ##      - An array or a seq
  ## Result:
  ##      - A view Tensor of the same shape
  return from_blob(
    oa[0].unsafeAddr,
    oa.len.int64,
    toScalarKind(T)
  ).clone()

func toTensor*[T: seq|array](oa: openarray[T]): Tensor =
  ## Interpret an openarray of openarray as a CPU Tensor
  ##
  ## Input:
  ##      - A nested array or a seq
  ## Result:
  ##      - A view Tensor of the same shape
  let shape = getShape(oa)
  type BaseType = getBaseType(T)

  result = empty(
    shape.asTorchView(),
    BaseType.toScalarKind()
  )

  let data = result.data_ptr(BaseType)
  for i, val in enumerate(flatIter(oa)):
    data[i] = val
