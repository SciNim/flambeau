import ../tensors
import ../raw/bindings/[rawtensors]
import ../raw/cpp/[std_cpp]
import ../raw/sugar/[rawinterop, indexing]
import std/[macros]

let t_dont_use_this {.used.} = initRawTensor()

## Aggregate
## -----------------------------------------------------------------------
# {.push noinit.}
# sum needs wrapper procs/templates to allow for using nim arrays and single axis.
func sum*[T](self: Tensor[T]): Tensor[T] =
  asTensor[T](
    rawtensors.sum(asRaw(self))
  )

func sum*[T](self: Tensor[T], dtype: ScalarKind): Tensor[T] =
  asTensor[T](
    rawtensors.sum(asRaw(self), dtype)
  )

func sum*[T](self: Tensor[T], axis: int64, keepdim: bool = false): Tensor[T] =
  asTensor[T](
    rawtensors.sum(asRaw(self), axis, keepdim)
  )

func sum*[T](self: Tensor[T], axis: int64, keepdim: bool = false, dtype: ScalarKind): Tensor[T] =
  asTensor[T](
    rawtensors.sum(asRaw(self), axis, keepdim, dtype)
  )

func sum*[T](self: Tensor[T], axis: openArray[int64], keepdim: bool = false): Tensor[T] =
  let axis = axis.asTorchView()
  asTensor[T](
    rawtensors.sum(asRaw(self), axis, keepdim)
  )

func sum*[T](self: Tensor[T], axis: openArray[int64], keepdim: bool = false, dtype: ScalarKind): Tensor[T] =
  let axis = axis.asTorchView()
  asTensor[T](
    rawtensors.sum(asRaw(self), axis, keepdim)
  )

# mean as well
func mean*[T](self: Tensor[T]): Tensor[T] =
  asTensor[T](
    rawtensors.mean(asRaw(self))
  )

func mean*[T](self: Tensor[T], dtype: ScalarKind): Tensor[T] =
  asTensor[T](
    rawtensors.mean(asRaw(self), dtype)
  )

func mean*[T](self: Tensor[T], axis: int64, keepdim: bool = false): Tensor[T] =
  asTensor[T](
    rawtensors.mean(asRaw(self), axis, keepdim)
  )

func mean*[T](self: Tensor[T], axis: int64, keepdim: bool = false, dtype: ScalarKind): Tensor[T] =
  asTensor[T](
    rawtensors.mean(asRaw(self), axis, keepdim, dtype)
  )

func mean*[T](self: Tensor[T], axis: openArray[int64], keepdim: bool = false): Tensor[T] =
  let axis = axis.asTorchView()
  asTensor[T](
    rawtensors.mean(asRaw(self), axis, keepdim)
  )

func mean*[T](self: Tensor[T], axis: openArray[int64], keepdim: bool = false, dtype: ScalarKind): Tensor[T] =
  let axis = axis.asTorchView()
  asTensor[T](
    rawtensors.mean(asRaw(self), axis, keepdim, dtype)
  )

# median requires std::tuple
func prod*[T](self: Tensor[T]): Tensor[T] =
  asTensor[T](
    rawtensors.prod(asRaw(self))
  )

func prod*[T](self: Tensor[T], dtype: ScalarKind): Tensor[T] =
  asTensor[T](
    rawtensors.prod(asRaw(self), dtype)
  )

func prod*[T](self: Tensor[T], axis: int64, keepdim: bool = false): Tensor[T] =
  asTensor[T](
    rawtensors.prod(asRaw(self), axis, keepdim)
  )

func prod*[T](self: Tensor[T], axis: int64, keepdim: bool = false, dtype: ScalarKind): Tensor[T] =
  asTensor[T](
    rawtensors.prod(asRaw(self), axis, keepdim, dtype)
  )

func min*[T](self: Tensor[T]): Tensor[T] =
  asTensor[T](
    rawtensors.min(asRaw(self))
  )

func min*[T](self: Tensor[T], axis: int64, keepdim: bool = false): tuple[values: Tensor[T], indices: Tensor[int]] =
  ## Returns a tuple (values, indices) of type (TensorT, TensorInt64)
  ## of the minimum values and their index in the specified axis
  let cppMinTuple = rawtensors.min(asRaw(self), axis, keepdim)
  result.values = asTensor[T](cppMinTuple.get(0))
  result.indices = asTensor[int](cppMinTuple.get(1))

func max*[T](self: Tensor[T]): Tensor[T] =
  asTensor[T](
    rawtensors.max(asRaw(self))
  )

func max*[T](self: Tensor[T], axis: int64, keepdim: bool = false): tuple[values: Tensor[T], indices: Tensor[int]] =
  ## Returns a tuple (values, indices) of type (TensorT, TensorInt64)
  ## of the maximum values and their index in the specified axis
  let cppMaxTuple = rawtensors.max(asRaw(self), axis, keepdim)
  result.values = asTensor[T](cppMaxTuple.get(0))
  result.indices = asTensor[int](cppMaxTuple.get(1))

func variance*[T](self: Tensor[T], unbiased: bool = true): Tensor[T] =
  asTensor[T](
    rawtensors.variance(asRaw(self), unbiased)
  )

func variance*[T](self: Tensor[T], axis: int64, unbiased: bool = true, keepdim: bool = false): Tensor[T] =
  asTensor[T](
    rawtensors.variance(asRaw(self), axis, unbiased, keepdim)
  )

func variance*[T](self: Tensor[T], axis: openArray[int64], unbiased: bool = true, keepdim: bool = false): Tensor[T] =
  let axis = axis.asTorchView()
  asTensor[T](
    rawtensors.variance(asRaw(self), axis, unbiased, keepdim)
  )

func stddev*[T](self: Tensor[T], unbiased: bool = true): Tensor[T] =
  asTensor[T](
    rawtensors.stddev(asRaw(self), unbiased)
  )

func stddev*[T](self: Tensor[T], axis: int64, unbiased: bool = true, keepdim: bool = false): Tensor[T] =
  asTensor[T](
    rawtensors.stddev(asRaw(self), axis, unbiased, keepdim)
  )

func stddev*[T](self: Tensor[T], axis: openArray[int64], unbiased: bool = true, keepdim: bool = false): Tensor[T] =
  let axis = axis.asTorchView()
  asTensor[T](
    rawtensors.stddev(asRaw(self), axis, unbiased, keepdim)
  )
