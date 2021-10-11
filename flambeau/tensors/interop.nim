import std/[complex, macros]
import ../raw/bindings/[rawtensors]
import ../raw/cpp/[std_cpp]
import ../raw/sugar/[indexing]
import ../tensors

import ../raw/sugar/interop as rawinterop
export rawinterop

func toTensorView*[T: SomeTorchType](oa: openArray[T]): lent Tensor[T] {.noinit.} =
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
  var res: RawTensor = toRawTensorView[T](oa)
  result = asTensor[T](res)

func toTensor*[T: SomeTorchType](oa: openArray[T]): Tensor[T] {.noinit.} =
  ## Interpret an openarray as a CPU Tensor
  ##
  ## Input:
  ##      - An array or a seq
  ## Result:
  ##      - A view Tensor of the same shape
  var res: RawTensor = toRawTensorFromScalar[T](oa)
  result = asTensor[T](res)

func toTensor*[T: seq|array](oa: openArray[T]): auto {.noinit.} =
  ## Interpret an openarray of openarray as a CPU Tensor
  ##
  ## Input:
  ##      - A nested array or a seq
  ## Result:
  ##      - A view Tensor of the same shape
  type V = getBaseType(T)
  var res: RawTensor = toRawTensorFromSeq(oa)
  result = asTensor[V](res)

proc `$`*[T](t: Tensor[T]): string =
  $(asRaw(t))
