import ../tensors
import ../raw/bindings/[rawtensors]
import ../raw/cpp/[std_cpp]
import ../raw/sugar/[interop, indexing]
import std/[macros]

## Aggregate
## -----------------------------------------------------------------------
{.push noinit.}
# sum needs wrapper procs/templates to allow for using nim arrays and single axis.
func sum*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.sum(convertRawTensor(self))
  )

func sum*[T](self: Tensor[T], dtype: ScalarKind): Tensor[T] =
  convertTensor[T](
    rawtensors.sum(convertRawTensor(self), dtype)
  )

func sum*[T](self: Tensor[T], axis: int64, keepdim: bool = false): Tensor[T] =
  convertTensor[T](
    rawtensors.sum(convertRawTensor(self), axis, keepdim)
  )

func sum*[T](self: Tensor[T], axis: int64, keepdim: bool = false, dtype: ScalarKind): Tensor[T] =
  convertTensor[T](
    rawtensors.sum(convertRawTensor(self), axis, keepdim, dtype)
  )

func sum*[T](self: Tensor[T], axis: openArray[int64], keepdim: bool = false): Tensor[T] =
  let axis = axis.asTorchView()
  convertTensor[T](
    rawtensors.sum(convertRawTensor(self), axis, keepdim)
  )

func sum*[T](self: Tensor[T], axis: openArray[int64], keepdim: bool = false, dtype: ScalarKind): Tensor[T] =
  let axis = axis.asTorchView()
  convertTensor[T](
    rawtensors.sum(convertRawTensor(self), axis, keepdim)
  )


# mean as well
func mean*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.mean(convertRawTensor(self))
  )

func mean*[T](self: Tensor[T], dtype: ScalarKind): Tensor[T] =
  convertTensor[T](
    rawtensors.mean(convertRawTensor(self), dtype)
  )

func mean*[T](self: Tensor[T], axis: int64, keepdim: bool = false): Tensor[T] =
  convertTensor[T](
    rawtensors.mean(convertRawTensor(self), axis, keepdim)
  )

func mean*[T](self: Tensor[T], axis: int64, keepdim: bool = false, dtype: ScalarKind): Tensor[T] =
  convertTensor[T](
    rawtensors.mean(convertRawTensor(self), axis, keepdim, dtype)
  )

func mean*[T](self: Tensor[T], axis: openArray[int64], keepdim: bool = false): Tensor[T] =
  let axis = axis.asTorchView()
  convertTensor[T](
    rawtensors.mean(convertRawTensor(self), axis, keepdim)
  )

func mean*[T](self: Tensor[T], axis: openArray[int64], keepdim: bool = false, dtype: ScalarKind): Tensor[T] =
  let axis = axis.asTorchView()
  convertTensor[T](
    rawtensors.mean(convertRawTensor(self), axis, keepdim, dtype)
  )

# median requires std::tuple

func prod*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.prod(convertRawTensor(self))
  )

func prod*[T](self: Tensor[T], dtype: ScalarKind): Tensor[T] =
  convertTensor[T](
    rawtensors.prod(convertRawTensor(self), dtype)
  )

func prod*[T](self: Tensor[T], axis: int64, keepdim: bool = false): Tensor[T] =
  convertTensor[T](
    rawtensors.prod(convertRawTensor(self), axis, keepdim)
  )

func prod*[T](self: Tensor[T], axis: int64, keepdim: bool = false, dtype: ScalarKind): Tensor[T] =
  convertTensor[T](
    rawtensors.prod(convertRawTensor(self), axis, keepdim, dtype)
  )

func min*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.min(convertRawTensor(self))
  )

func min*[T](self: Tensor[T], axis: int64, keepdim: bool = false): tuple[values: Tensor[T], indices: Tensor[int]] =
  ## Returns a tuple (values, indices) of type (TensorT, TensorInt64)
  ## of the minimum values and their index in the specified axis
  let cppMinTuple = rawtensors.min(convertRawTensor(self), axis, keepdim)
  result.values = convertTensor[T](cppMinTuple.get(0))
  result.indices = convertTensor[int](cppMinTuple.get(1))

func max*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.max(convertRawTensor(self))
  )


func max*[T](self: Tensor[T], axis: int64, keepdim: bool = false): tuple[values: Tensor[T], indices: Tensor[int]] =
  ## Returns a tuple (values, indices) of type (TensorT, TensorInt64)
  ## of the maximum values and their index in the specified axis
  let cppMaxTuple = rawtensors.max(convertRawTensor(self), axis, keepdim)
  result.values = convertTensor[T](cppMaxTuple.get(0))
  result.indices = convertTensor[int](cppMaxTuple.get(1))


func variance*[T](self: Tensor[T], unbiased: bool = true): Tensor[T] =
  convertTensor[T](
    rawtensors.variance(convertRawTensor(self), unbiased)
  )

func variance*[T](self: Tensor[T], axis: int64, unbiased: bool = true, keepdim: bool = false): Tensor[T] =
  convertTensor[T](
    rawtensors.variance(convertRawTensor(self), axis, unbiased, keepdim)
  )

func variance*[T](self: Tensor[T], axis: openArray[int64], unbiased: bool = true, keepdim: bool = false): Tensor[T] =
  let axis = axis.asTorchView()
  convertTensor[T](
    rawtensors.variance(convertRawTensor(self), axis, unbiased, keepdim)
  )

func stddev*[T](self: Tensor[T], unbiased: bool = true): Tensor[T] =
  convertTensor[T](
    rawtensors.stddev(convertRawTensor(self), unbiased)
  )

func stddev*[T](self: Tensor[T], axis: int64, unbiased: bool = true, keepdim: bool = false): Tensor[T] =
  convertTensor[T](
    rawtensors.stddev(convertRawTensor(self), axis, unbiased, keepdim)
  )

func stddev*[T](self: Tensor[T], axis: openArray[int64], unbiased: bool = true, keepdim: bool = false): Tensor[T] =
  let axis = axis.asTorchView()
  convertTensor[T](
    rawtensors.stddev(convertRawTensor(self), axis, unbiased, keepdim)
  )

