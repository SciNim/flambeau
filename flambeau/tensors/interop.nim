import std/[complex, macros, enumerate, strformat]
import ../raw/bindings/[rawtensors]
import ../raw/cpp/[std_cpp]
import ../raw/sugar/[indexing]
import ../raw/private/dynamic_stack_arrays
import ../tensors

import ../raw/sugar/rawinterop
export rawinterop

let t_dont_use_this {.used.} = initRawTensor()

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
  return from_blob[T](
    arg[0].unsafeAddr,
    arg.len.int64,
  )

func toTensorFromScalar*[T: SomeTorchType](arg: openarray[T]): Tensor[T] =
  ## Interpret an openarray as a CPU Tensor
  ##
  ## Input:
  ##      - An array or a seq
  ## Result:
  ##      - A view Tensor of the same shape
  # Don't know why clone is segfaulting so it seems easier to just copyMem the openArray
  let shape = getShape(arg).toSeq()
  result = empty[T](shape)
  let memlen = (arg.len()*sizeof(T)) div sizeof(byte)
  let data = result.data_ptr()
  copyMem(data, arg[0].unsafeAddr, memlen)

func toTensorFromSeq*[T: seq|array, U](oa: openarray[T]): Tensor[U] =
  ## Interpret an openarray of openarray as a CPU Tensor
  ##
  ## Input:
  ##      - A nested array or a seq
  ## Result:
  ##      - A view Tensor of the same shape
  let shape = getShape(oa).toSeq()
  result = empty[U](shape)
  let data = result.data_ptr()
  for i, val in enumerate(flatIter(oa)):
    data[i] = val


func toTensor*[T: SomeTorchType](oa: openArray[T]): Tensor[T]  =
  ## Interpret an openarray as a CPU Tensor
  ##
  ## Input:
  ##      - An array or a seq
  ## Result:
  ##      - A view Tensor of the same shape
  result = toTensorFromScalar(oa)

func toTensor*[T: seq|array](oa: openArray[T]): auto  =
  ## Interpret an openarray of openarray as a CPU Tensor
  ##
  ## Input:
  ##      - A nested array or a seq
  ## Result:
  ##      - A view Tensor of the same shape
  type V = getBaseType(T)
  result = toTensorFromSeq[T, V](oa)
  
proc `$`*[T](t: Tensor[T]): string =
  "Tensor[" & $T & "]\n" & $(toCppString(asRaw(t)))