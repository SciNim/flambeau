import std/[complex, macros]
import ../raw/bindings/[rawtensors]
import ../raw/cpp/[std_cpp]
import ../raw/sugar/[indexing]
import ../tensors

import ../raw/sugar/interop as rawinterop
export rawinterop

func toTensorView*[T: SomeTorchType](oa: openArray[T]): lent Tensor[T] =
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
  #
  # return tensors.from_blob[T](
  #   oa[0].unsafeAddr,
  #   oa.len.int64,
  #   T
  # )
  var res : RawTensor = toRawTensorView[T](oa)
  return convertTensor[T](res)

func toTensor*[T: SomeTorchType](oa: openArray[T]): Tensor[T] =
  ## Interpret an openarray as a CPU Tensor
  ##
  ## Input:
  ##      - An array or a seq
  ## Result:
  ##      - A view Tensor of the same shape
  # let shape = getShape(oa)
  # result = empty[T](shape.asTorchView())
  #
  # return from_blob[T](
  #   oa[0].unsafeAddr,
  #   oa.len.int64,
  # ).clone()
  #
  var res : RawTensor = toRawTensor[T](oa)
  return convertTensor[T](res)


func toTensor*[T: seq|array](oa: openArray[T]): auto =
  ## Interpret an openarray of openarray as a CPU Tensor
  ##
  ## Input:
  ##      - A nested array or a seq
  ## Result:
  ##      - A view Tensor of the same shape
  type BaseType = getBaseType(T)
  var res : RawTensor = toRawTensorFromSeq(oa)
  return convertTensor[BaseType](res)

macro `[]`*[T](t: Tensor[T], args: varargs[untyped]): untyped =
  result = quote do:
    [](`t.raw`, args)

macro `[]=`*[T](t: var Tensor[T], args: varargs[untyped]): untyped =
  result = quote do:
    [] = (`t.raw`, args)

proc `$`*[T](t: Tensor[T]): string =
  # `$`(convertRawTensor(t))
  "Tensor\n" & `$`(toCppString(t.raw))
