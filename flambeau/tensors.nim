import raw/bindings/[rawtensors, c10]
import raw/cpp/[std_cpp]
import raw/sugar/rawinterop
import raw/sugar/indexing
import std/[complex, macros]

export SomeTorchType
export Torch
# export hasCuda
export cuda_is_available

{.experimental: "views".} # TODO

type
  Tensor*[T] = object
    raw: RawTensor

  ## If Tensor is used as a field of an object, it has to be used with {.noinit.} pragma
  ## This is because torch::Tensor is a hidden intrusive_ptr<TensorImpl> and the zeroMem() done by Nim compiler resets the ref count fields,
  ## which triggers a call to torch::Tensor::reset()

template asRaw*[T: SomeTorchType](t: Tensor[T]): RawTensor =
  # RawTensor(t)
  t.raw

template asRaw*[T: SomeTorchType](t: var Tensor[T]): var RawTensor =
  # RawTensor(t)
  t.raw

template asTensor*[T: SomeTorchType](t: RawTensor): Tensor[T] =
  # if T is complex then T = Complex32 gets convertes to kComplexF32 by converter
  # Tensor[T](to(t, typedesc[T]))
  var tensor: Tensor[T]
  tensor.raw = to(t, typedesc[T])
  tensor

proc initTensor*[T](): Tensor[T] =
  asRaw(result) = initRawTensor()

proc initTensor*[T](a: Tensor[T]): Tensor[T] =
  asRaw(result) = initRawTensor(asRaw(a))

#{.push inline.}

# Strings & Debugging
# -----------------------------------------------------------------------

proc print*[T](self: Tensor[T]) {.sideeffect.} =
  print(asRaw(self))

# Metadata
# -----------------------------------------------------------------------

func dim*[T](self: Tensor[T]): int64 =
  ## Number of dimensions
  dim(asRaw(self))

func reset*[T](self: var Tensor[T]) =
  reset(asRaw(self))

func is_same*[T](self, other: Tensor[T]): bool =
  ## Reference equality
  ## Do the tensors use the same memory.
  is_same(asRaw(self), asRaw(other))

func sizes*[T](self: Tensor[T]): IntArrayRef =
  ## This is Arraymancer and Numpy "shape"
  sizes(asRaw(self))

func ndimension*[T](self: Tensor[T]): int64 =
  ## This is Arraymancer rank
  ndimension(asRaw(self))

func rank*[T](self: Tensor[T]): int64 =
  ##  For arraymancer compatibility
  ndimension[T](self)

func shape*[T](self: Tensor[T]): seq[int64] =
  ## This is Arraymancer and Numpy "shape"
  let tmpshape = sizes(self)
  let r = self.ndimension()
  result = newSeq[int64](r)
  for i in 0 ..< r:
    result[i] = tmpshape[i]

func strides*[T](self: Tensor[T]): seq[int64] =
  let tmp = strides(asRaw(self))
  let r = self.ndimension()
  result = newSeq[int64](r)
  for i in 0 ..< r:
    result[i] = tmp[i]

func nbytes*[T](self: Tensor[T]): uint =
  ## Bytes-size of the Tensor
  nbytes(asRaw(self))

func numel*[T](self: Tensor[T]): int64 =
  ## This is Arraymancer and Numpy "size"
  numel(asRaw(self))

func size*[T](self: Tensor[T], axis: int64): int64 =
  size(asRaw(self))

func itemsize*[T](self: Tensor[T]): uint =
  itemsize(asRaw(self))

func element_size*[T](self: Tensor[T]): int64 =
  element_size(asRaw(self))

# Accessors
# -----------------------------------------------------------------------

func data_ptr*[T](self: Tensor[T]): ptr UncheckedArray[T] =
  ## Gives raw access to a tensor data of type T.
  ##
  ## This is a very low-level procedure. You need to take care
  ## of the tensor shape and strides yourself.
  ##
  ## It is recommended to use this only on contiguous tensors
  ## (freshly created or freshly cloned) and to avoid
  ## sliced tensors.
  when T is byte | uint8 | SomeSignedInt | SomeFloat:
    data_ptr(asRaw(self), T)
  elif T is Complex32:
    cast[ptr UncheckedArray[Complex32]](data_ptr(asRaw(self), C10_Complex[float32]))
  elif T is Complex64:
    cast[ptr UncheckedArray[Complex64]](data_ptr(asRaw(self), C10_Complex[float64]))

# Backend
# -----------------------------------------------------------------------

func has_storage*[T](self: Tensor[T]): bool =
  has_storage(asRaw(self))

func get_device*[T](self: Tensor[T]): int64 =
  get_device(asRaw(self))

func is_cuda*[T](self: Tensor[T]): bool =
  is_cuda(asRaw(self))

func is_hip*[T](self: Tensor[T]): bool =
  is_hip(asRaw(self))

func is_sparse*[T](self: Tensor[T]): bool =
  is_sparse(asRaw(self))

func is_mkldnn*[T](self: Tensor[T]): bool =
  is_mkldnn(asRaw(self))

func is_vulkan*[T](self: Tensor[T]): bool =
  is_vulkan(asRaw(self))

func is_quantized*[T](self: Tensor[T]): bool =
  is_quantized(asRaw(self))

func is_meta*[T](self: Tensor[T]): bool =
  is_meta(asRaw(self))

func cpu*[T](self: Tensor[T]): Tensor[T] =
  asTensor[T](cpu(asRaw(self)))

func cuda*[T](self: Tensor[T]): Tensor[T] =
  asTensor[T](cuda(asRaw(self)))

func hip*[T](self: Tensor[T]): Tensor[T] =
  asTensor[T](hip(asRaw(self)))

func vulkan*[T](self: Tensor[T]): Tensor[T] =
  asTensor[T](vulkan(asRaw(self)))

func to*[T](self: Tensor[T], device: DeviceKind): Tensor[T] =
  asTensor[T](to(asRaw(self), device))

func to*[T](self: Tensor[T], device: Device): Tensor[T] =
  asTensor[T](to(asRaw(self), device))

# dtype
# -----------------------------------------------------------------------
func to*[T](self: Tensor[T], dtype: typedesc[SomeTorchType]): Tensor[dtype] =
  # Use typedesc -> ScalarKind converter here : for T = Complex32 T is converted to kComplexF32
  asTensor[dtype](rawtensors.to(asRaw(self), dtype))

func scalarType*[T](self: Tensor[T]): typedesc =
  toTypedesc(rawtensors.scalarType(asRaw(self)))

# Constructors
# -----------------------------------------------------------------------
# DeviceType and ScalarType are auto-convertible to TensorOptions

func from_blob*[T](data: pointer, sizes: openArray[int64], options: TensorOptions | DeviceKind): Tensor[T] =
  let dims = sizes.asTorchView
  asTensor[T](rawtensors.from_blob(data, dims, options))

func from_blob*[T](data: pointer, sizes: openArray[int64]): Tensor[T] =
  let dims = sizes.asTorchView
  asTensor[T](rawtensors.from_blob(data, dims, T))

func from_blob*[T](data: pointer, sizes: int64, options: TensorOptions | DeviceKind): Tensor[T] =
  asTensor[T](rawtensors.from_blob(data, sizes, options))

func from_blob*[T](data: pointer, sizes: int64): Tensor[T] =
  asTensor[T](rawtensors.from_blob(data, sizes, T))

func from_blob*[T](data: pointer, sizes, strides: openArray[int64], options: TensorOptions | DeviceKind): Tensor[T] =
  let
    dims = sizes.asTorchView
    stridest = strides.asTorchView
  asTensor[T](rawtensors.from_blob(data, dims, stridest, options))

func from_blob*[T](data: pointer, sizes, strides: openArray[int64]): Tensor[T] =
  let
    dims = sizes.asTorchView
    stridest = strides.asTorchView
  asTensor[T](rawtensors.from_blob(data, dims, stridest, T))

func empty*[T](size: openArray[int64], options: TensorOptions | DeviceKind): Tensor[T] =
  ## Create an uninitialized tensor of shape `size`
  ## The tensor data must be filled manually
  ##
  ## The output tensor will be row major (C contiguous)
  let dims = size.asTorchView()
  asTensor[T](rawtensors.empty(dims, options))

func empty*[T](size: openArray[int64]): Tensor[T] =
  let dims = size.asTorchView()
  asTensor[T](rawtensors.empty(dims, T))

func clone*[T](self: Tensor[T]): Tensor[T] =
  asTensor[T](rawtensors.clone(asRaw(self)))

# TODO : make these works for complex helper
# func view_as_real*[T: SomeFloat](self: Tensor[Complex[T]]) : Tensor[T] =
#   asTensor[T](
#     view_as_real(asRaw(self))
#   )
# func view_as_complex*[T: SomeFloat](self: Tensor[T]) : Tensor[Complex[T]] =
#   asTensor[Complex[T]](
#     view_as_complex(asRaw(self))
#   )

# Random sampling
# -----------------------------------------------------------------------

func random_mut*[T](self: var Tensor[T], start, stopEx: int64) =
  random_mut(asRaw(self), start, stopEx)

func randint*[T](start, stopEx: int64, args: varargs): Tensor[T] =
  asTensor[T](rawtensors.randint(start, stopEx, args))

func randint*[T](start, stopEx: int64, size: openArray[int64]): Tensor[T] =
  let dims = size.asTorchView()
  asTensor[T](rawtensors.randint(start, stopEx, dims))

func rand_like*[T](self: Tensor[T], options: TensorOptions | DeviceKind | Device): Tensor[T] =
  asTensor[T](rawtensors.rand_like(asRaw(self), options))

func rand_like*[T](self: Tensor[T]): Tensor[T] =
  asTensor[T](rawtensors.rand_like(asRaw(self), T))

func rand*[T](size: openArray[int64]): Tensor[T] =
  let dims = size.asTorchView()
  asTensor[T](rawtensors.rand(dims))

# Shapeshifting
# -----------------------------------------------------------------------

func reshape*[T](self: Tensor[T], size: openArray[int64]): Tensor[T] =
  let dims = size.asTorchView()
  asTensor[T](reshape(asRaw(self), dims))

func view*[T](self: Tensor[T], size: openArray[int64]): Tensor[T] =
  let dims = size.asTorchView()
  asTensor[T](reshape(asRaw(self), dims))

func transpose*[T](self: Tensor[T], dim0, dim1: int64): Tensor[T] =
  ## Swaps two dimensions. Returns a tensor that is a transposed version of input.
  ## The given dimensions dim0 and dim1 are swapped.
  asTensor[T](rawtensors.transpose(asRaw(self), dim0, dim1))

func t*[T](self: Tensor[T]): Tensor[T] =
  ## Transposes a 2D tensor. Equivalent to transpose(0, 1).
  ## This function is only supported for 2D tensors.
  asTensor[T](rawtensors.t(asRaw(self)))

func permute*[T](self: Tensor[T], dims: openArray[int64]): Tensor[T] =
  ## Returns a view of the original tensor with its dimensions permuted.
  let dimsList = dims.asTorchView()
  asTensor[T](rawtensors.permute(asRaw(self), dimsList))

# Automatic Differentiation
# -----------------------------------------------------------------------

func backward*[T](self: var Tensor[T]) =
  backward(asRaw(self))

func detach*[T](self: Tensor[T]): Tensor[T] =
  ## Returns a new tensor, detached from the current computation graph.
  ## The result will never require gradient.
  asTensor[T](rawtensors.detach(asRaw(self)))

# # Functions.h
# # -----------------------------------------------------------------------
func contiguous*[T](self: Tensor[T]): Tensor[T] =
  asTensor[T](rawtensors.contiguous(asRaw(self)))

func toType*[T](self: Tensor[T], dtype: ScalarKind): Tensor[T] =
  asTensor[T](rawtensors.toType(asRaw(self), dtype))

func toSparse*[T](self: Tensor[T]): Tensor[T] =
  asTensor[T](rawtensors.toSparse(asRaw(self)))

func toSparse*[T](self: Tensor[T], sparseDim: int64): Tensor[T] =
  asTensor[T](rawtensors.toSparse(asRaw(self), sparseDim))

func eye*[T](n: int64): Tensor[T] =
  asTensor[T](rawtensors.eye(n, T))

func eye*[T](n: int64, options: DeviceKind | TensorOptions): Tensor[T] =
  asTensor[T](rawtensors.eye(n, options))

func zeros*[T](dim: int64): Tensor[T] =
  asTensor[T](rawtensors.zeros(dim))

func zeros*[T](dim: openArray[int64]): Tensor[T] =
  let dims = dim.asTorchView()
  asTensor[T](rawtensors.zeros(dims, T))

func zeros*[T](dim: openArray[int64], options: DeviceKind | TensorOptions): Tensor[T] =
  let dims = dim.asTorchView()
  asTensor[T](rawtensors.zeros(dims, options))

func ones*[T](dim: int64): Tensor[T] =
  asTensor[T](rawtensors.ones(dim))

func ones*[T](dim: openArray[int64]): Tensor[T] =
  let dims = dim.asTorchView()
  asTensor[T](rawtensors.ones(dims, T))

func ones*[T](dim: openArray[int64], options: DeviceKind | TensorOptions): Tensor[T] =
  let dims = dim.asTorchView()
  asTensor[T](rawtensors.ones(dims, options))

func full*[T](size: openArray[int64], fill_value: T): Tensor[T] =
  let dims = size.asTorchView()
  asTensor[T](rawtensors.full(dims, fill_value, T))

func full*[T](size: openArray[int64], fill_value: T, options: DeviceKind | TensorOptions): Tensor[T] =
  let dims = size.asTorchView()
  asTensor[T](rawtensors.full(dims, fill_value, options))

func randn*[T](size: openArray[int64]): Tensor[T] =
  let dims = size.asTorchView()
  asTensor[T](rawtensors.randn(dims, T))

func randn*[T](size: openArray[int64], options: DeviceKind | TensorOptions): Tensor[T] =
  let dims = size.asTorchView()
  asTensor[T](rawtensors.randn(dims, options))

func linspace*[T](start, stop: T, steps: int64, options: TensorOptions): Tensor[T] =
  asTensor[T](rawtensors.linspace(start, stop, steps, options))

func linspace*[T](start, stop: T, steps: int64, options: DeviceKind): Tensor[T] =
  asTensor[T](rawtensors.linspace(start, stop, steps, options))

func linspace*[T](start, stop: T, steps: int64, options: Device): Tensor[T] =
  asTensor[T](rawtensors.linspace(start, stop, steps, options))

func linspace*[T](start, stop: T, steps: int64): Tensor[T] =
  asTensor[T](rawtensors.linspace(start, stop, steps, T))

func linspace*[T](start, stop: T): Tensor[T] =
  asTensor[T](rawtensors.linspace(start, stop))

func logspace*[T](start, stop: T, steps: int64, options: TensorOptions): Tensor[T] =
  asTensor[T](rawtensors.logspace(start, stop, steps, options))

func logspace*[T](start, stop: T, steps: int64, options: DeviceKind): Tensor[T] =
  asTensor[T](rawtensors.logspace(start, stop, steps, options))

func logspace*[T](start, stop: T, steps: int64, options: Device): Tensor[T] =
  asTensor[T](rawtensors.logspace(start, stop, steps, options))

func logspace*[T](start, stop: T, steps: int64): Tensor[T] =
  asTensor[T](rawtensors.logspace(start, stop, steps, T))

func logspace*[T](start, stop: T): Tensor[T] =
  asTensor[T](rawtensors.logspace(start, stop))

func arange*[T](stop: T, options: TensorOptions): Tensor[T] =
  asTensor[T](rawtensors.arange(stop, options))

func arange*[T](stop: T, options: DeviceKind): Tensor[T] =
  asTensor[T](rawtensors.arange(stop, options))

func arange*[T](stop: T, options: Device): Tensor[T] =
  asTensor[T](rawtensors.arange(stop, options))

func arange*[T](stop: T): Tensor[T] =
  asTensor[T](rawtensors.arange(stop, T))

func arange*[T](start, stop: T, options: TensorOptions): Tensor[T] =
  asTensor[T](rawtensors.arange(start, stop, options))
func arange*[T](start, stop: T, options: DeviceKind): Tensor[T] =
  asTensor[T](rawtensors.arange(start, stop, options))
func arange*[T](start, stop: T, options: Device): Tensor[T] =
  asTensor[T](rawtensors.arange(start, stop, options))
func arange*[T](start, stop: T): Tensor[T] =
  asTensor[T](rawtensors.arange(start, stop, T))

func arange*[T](start, stop, step: T, options: TensorOptions): Tensor[T] =
  asTensor[T](rawtensors.arange(start, stop, step, options))
func arange*[T](start, stop, step: T, options: DeviceKind): Tensor[T] =
  asTensor[T](rawtensors.arange(start, stop, step, options))
func arange*[T](start, stop, step: T, options: Device): Tensor[T] =
  asTensor[T](rawtensors.arange(start, stop, step, options))
func arange*[T](start, stop, step: T): Tensor[T] =
  asTensor[T](rawtensors.arange(start, stop, step, T))

# Operations
# -----------------------------------------------------------------------
func add*[T](self: Tensor[T], other: Tensor[T], alpha: Scalar = 1): Tensor[T] =
  asTensor[T](rawtensors.add(self, other, alpha))
func add*[T](self: Tensor[T], other: Scalar, alpha: Scalar = 1): Tensor[T] =
  asTensor[T](rawtensors.add(self, other, alpha))

func addmv*[T](self: Tensor[T], mat: Tensor[T], vec: Tensor[T], beta: Scalar = 1, alpha: Scalar = 1): Tensor[T] =
  asTensor[T](rawtensors.addmv(self, mat, vec, beta, alpha))

func addmm*[T](t, mat1, mat2: Tensor[T], beta: Scalar = 1, alpha: Scalar = 1): Tensor[T] =
  asTensor[T](rawtensors.addmm(t, mat1, mat2, beta, alpha))

func mm*[T](t, other: Tensor[T]): Tensor[T] =
  asTensor[T](rawtensors.mm(asRaw(t), asRaw(other)))

func matmul*[T](t, other: Tensor[T]): Tensor[T] =
  asTensor[T](rawtensors.matmul(asRaw(t), asRaw(other)))

func bmm*[T](t, other: Tensor[T]): Tensor[T] =
  asTensor[T](rawtensors.bmm(asRaw(t), asRaw(other)))

func luSolve*[T](t, data, pivots: Tensor[T]): Tensor[T] =
  asTensor[T](rawtensors.luSolve(asRaw(t), asRaw(data), asRaw(pivots)))

func qr*[T](self: Tensor[T], some: bool = true): tuple[q: Tensor[T], r: Tensor[T]] =
  ## Returns a tuple:
  ## - Q of shape (∗,m,k)
  ## - R of shape (∗,k,n)
  ## with k=min(m,n) if some is true otherwise k=m
  ##
  ## The QR decomposition is batched over dimension(s) *[T]
  ## t = QR
  let cppTupRes = rawtensors.qr(asRaw(self), some)
  result.q = asTensor[T](cppTupRes.get(0))
  result.r = asTensor[T](cppTupRes.get(1))

# addr?
func all*[T](self: Tensor[T], axis: int64): Tensor[T] =
  asTensor[T](rawtensors.all(asRaw(self), axis))
func all*[T](self: Tensor[T], axis: int64, keepdim: bool): Tensor[T] =
  asTensor[T](rawtensors.all(asRaw(self), axis, keepdim))

func any*[T](self: Tensor[T], axis: int64): Tensor[T] =
  asTensor[T](rawtensors.any(asRaw(self), axis))
func any*[T](self: Tensor[T], axis: int64, keepdim: bool): Tensor[T] =
  asTensor[T](rawtensors.any(asRaw(self), axis, keepdim))

func argmax*[T](self: Tensor[T]): Tensor[int] =
  asTensor[int](rawtensors.argmax(asRaw(self)))
func argmax*[T](self: Tensor[T], axis: int64, keepdim: bool = false): Tensor[int] =
  asTensor[int](rawtensors.argmax(asRaw(self), axis, keepdim))

func argmin*[T](self: Tensor[T]): Tensor[int] =
  asTensor[int](rawtensors.argmin(asRaw(self)))
func argmin*[T](self: Tensor[T], axis: int64, keepdim: bool = false): Tensor[int] =
  asTensor[int](rawtensors.argmin(asRaw(self), axis, keepdim))

func allClose*[T](t, other: Tensor[T], rtol: float64 = 1e-5, abstol: float64 = 1e-8, equalNan: bool = false): bool =
  allClose(asRaw(t), asRaw(other), rtol, abstol, equalNan)

import ./tensors/fft
export fft
import ./tensors/mathalgo
export mathalgo
import ./tensors/operators
export operators
import ./tensors/aggregate
export aggregate
import ./tensors/interop
export interop
import ./tensors/accessors
export accessors
import ./tensors/fancy_index
export fancy_index

# #func convolution*[T](self: Tensor, weight: Tensor, bias: Tensor, stride, padding, dilation: int64, transposed: bool, outputPadding: int64, groups: int64): Tensor
