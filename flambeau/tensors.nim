import raw/bindings/[rawtensors, c10]
import raw/cpp/[std_cpp]
import raw/sugar/[interop, indexing]
import std/[complex, macros]

{.experimental: "views".} # TODO

type
  Tensor*[T] {.pure, final.} = object

proc convertRawTensor[T](t: Tensor[T]) : RawTensor =
  result = cast[ptr RawTensor](unsafeAddr(t))[]

proc convertTensor[T](t: RawTensor) : Tensor[T] =
  result = cast[ptr Tensor[T]](unsafeAddr(t))[]

func toTensorView*[T: SomeTorchType](oa: openArray[T]): lent Tensor[T] =
  convertTensor[T](toRawTensorView[T](oa))

func toTensor*[T: SomeTorchType](oa: openArray[T]): Tensor[T] =
  convertTensor[T](toRawTensor[T](oa))

func toTensor*[T: seq|array](oa: openArray[T]): auto =
  # Get underlying type
  type U = getBaseType(T)
  # Ambiguous because of auto ?
  let res = toRawTensorFromSeq[T](oa)
  result = convertTensor[U](res)

macro `[]`*[T](t: Tensor[T], args: varargs[untyped]): untyped =
  result = quote do:
    [](`convertRawTensor(t)`, args)

macro `[]=`*[T](t: var Tensor[T], args: varargs[untyped]): untyped =
  result = quote do:
    [] = (`convertRawTensor(t)`, args)

proc `$`*[T](t: Tensor[T]): string =
  result = "Tensor\n" & $(toCppString(convertRawTensor(t)))

# Strings & Debugging
# -----------------------------------------------------------------------

proc print*[T](self: Tensor[T]) {.sideeffect, inline.} =
  print(convertRawTensor(self))

# Metadata
# -----------------------------------------------------------------------

func dim*[T](self: Tensor[T]): int64 {.inline.} =
  ## Number of dimensions
  dim(convertRawTensor(self))

func reset*[T](self: var Tensor[T]) {.inline.} =
  reset(convertRawTensor(self))

func is_same*[T](self, other: Tensor[T]): bool {.inline.} =
  ## Reference equality
  ## Do the tensors use the same memory.
  is_same(convertRawTensor(self), convertRawTensor(other))

func sizes*[T](self: Tensor[T]): IntArrayRef {.inline.} =
  ## This is Arraymancer and Numpy "shape"
  sizes(convertRawTensor(self))

func shape*[T](self: Tensor[T]): seq[int64] {.inline.} =
  ## This is Arraymancer and Numpy "shape"
  result = asNimView(sizes(convertRawTensor(self)))

func strides*[T](self: Tensor[T]): openArray[int64] {.inline.} =
  strides(convertRawTensor(self))

func ndimension*[T](self: Tensor[T]): int64 {.inline.} =
  ## This is Arraymancer rank
  ndimension(convertRawTensor(self))

func nbytes*[T](self: Tensor[T]): uint {.inline.} =
  ## Bytes-size of the Tensor
  nbytes(convertRawTensor(self))

func numel*[T](self: Tensor[T]): int64 {.inline.} =
  ## This is Arraymancer and Numpy "size"
  numel(convertRawTensor(self))

func size*[T](self: Tensor[T], axis: int64): int64 {.inline.} =
  size(convertRawTensor(self))

func itemsize*[T](self: Tensor[T]): uint {.inline.} =
  itemsize(convertRawTensor(self))

func element_size*[T](self: Tensor[T]): int64 {.inline.} =
  element_size(convertRawTensor(self))

# Accessors
# -----------------------------------------------------------------------

func data_ptr*[T](self: Tensor[T]): ptr UncheckedArray[T] {.inline.} =
  ## Gives raw access to a tensor data of type T.
  ##
  ## This is a very low-level procedure. You need to take care
  ## of the tensor shape and strides yourself.
  ##
  ## It is recommended to use this only on contiguous tensors
  ## (freshly created or freshly cloned) and to avoid
  ## sliced tensors.
  data_ptr(convertRawTensor(self), T)

# Backend
# -----------------------------------------------------------------------

func has_storage*[T](self: Tensor[T]): bool {.inline.} =
  has_storage(convertRawTensor(self))

func get_device*[T](self: Tensor[T]): int64 {.inline.} =
  get_device(convertRawTensor(self))

func is_cuda*[T](self: Tensor[T]): bool {.inline.} =
  is_cuda(convertRawTensor(self))

func is_hip*[T](self: Tensor[T]): bool {.inline.} =
  is_hip(convertRawTensor(self))

func is_sparse*[T](self: Tensor[T]): bool {.inline.} =
  is_sparse(convertRawTensor(self))

func is_mkldnn*[T](self: Tensor[T]): bool {.inline.} =
  is_mkldnn(convertRawTensor(self))

func is_vulkan*[T](self: Tensor[T]): bool {.inline.} =
  is_vulkan(convertRawTensor(self))

func is_quantized*[T](self: Tensor[T]): bool {.inline.} =
  is_quantized(convertRawTensor(self))

func is_meta*[T](self: Tensor[T]): bool {.inline.} =
  is_meta(convertRawTensor(self))

func cpu*[T](self: Tensor[T]): Tensor[T] {.inline.} =
  cpu(convertRawTensor(self))

func cuda*[T](self: Tensor[T]): Tensor[T] {.inline.} =
  cuda(convertRawTensor(self))

func hip*[T](self: Tensor[T]): Tensor[T] {.inline.} =
  hip(convertRawTensor(self))

func vulkan*[T](self: Tensor[T]): Tensor[T] {.inline.} =
  vulkan(convertRawTensor(self))

func to*[T](self: Tensor[T], device: DeviceKind): Tensor[T] {.inline.} =
  to(convertRawTensor(self), device)

func to*[T](self: Tensor[T], device: Device): Tensor[T] {.inline.} =
  to(convertRawTensor(self), device)

# dtype
# -----------------------------------------------------------------------
func to*[T](self: Tensor[T], dtype: typedesc[SomeTorchType]): Tensor[T] {.inline.} =
  # Use typedesc -> ScalarKind converter
  to(convertRawTensor(self), dtype)

func scalarType*[T](self: Tensor[T]): typedesc[T] {.inline.} =
  T

# Constructors
# -----------------------------------------------------------------------
# DeviceType and ScalarType are auto-convertible to TensorOptions

func init*[T](t: type Tensor[T]): Tensor[T] {.inline.} =
  init(convertRawTensor(t))

func from_blob*[T](data: pointer, sizes: openArray[int64], options: TensorOptions|DeviceKind): Tensor[T] {.inline.} =
  let sizes = sizes.asTorchView
  from_blob(data, sizes, options).Tensor[T]

func from_blob*[T](data: pointer, sizes: openArray[int64]): Tensor[T] {.inline.} =
  let sizes = sizes.asTorchView
  from_blob(data, sizes, T).Tensor[T]

func from_blob*[T](data: pointer, sizes: int64, options: TensorOptions|DeviceKind): Tensor[T] {.inline.} =
  from_blob(data, sizes, options).Tensor[T]

func from_blob*[T](data: pointer, sizes: int64): Tensor[T] {.inline.} =
  from_blob(data, sizes, T).Tensor[T]

func from_blob*[T](data: pointer, sizes, strides: openArray[int64], options: TensorOptions|DeviceKind): Tensor[T] {.inline.} =
  let
    sizes = sizes.asTorchView
    strides = strides.asTorchView
  from_blob(data, sizes, strides, options).Tensor[T]

func from_blob*[T](data: pointer, sizes, strides: openArray[int64]): Tensor[T] {.inline.} =
  let
    sizes = sizes.asTorchView
    strides = strides.asTorchView
  from_blob(data, sizes, strides, T).Tensor[T]

func empty*[T](size: IntArrayRef, options: TensorOptions|DeviceKind): Tensor[T] {.inline.} =
  ## Create an uninitialized tensor of shape `size`
  ## The tensor data must be filled manually
  ##
  ## The output tensor will be row major (C contiguous)
  let size = size.asTorchView
  empty(size, options).Tensor[T]

func empty*[T](size: IntArrayRef): Tensor[T] {.inline.} =
  let size = size.asTorchView
  empty(size, T).Tensor[T]

func clone*[T](self: Tensor[T]): Tensor[T] {.inline.} =
  clone(convertRawTensor(self)).Tensor[T]

# Random sampling
# -----------------------------------------------------------------------

func random_mut*[T](self: var Tensor[T], start, stopEx: int64) {.inline.} =
  random_mut(convertRawTensor(self), start, stopEx)

func randint*[T](start, stopEx: int64, args: varargs): Tensor[T] {.inline.} =
  randint(start, stopEx, args).Tensor[T]

func randint*[T](start, stopEx: int64, size: openArray[int64]): Tensor[T] {.inline.} =
  let size = size.asTorchView()
  randint(start, stopEx, size).Tensor[T]

func rand_like*[T](self: Tensor[T], options: TensorOptions|DeviceKind|Device): Tensor[T] {.inline.} =
  rand_like(convertRawTensor(self), options).Tensor[T]

func rand_like*[T](self: Tensor[T]): Tensor[T] {.inline.} =
  rand_like(convertRawTensor(self), T).Tensor[T]

func rand*[T](size: openArray[int64]): Tensor[T] {.inline.} =
  let size = size.asTorchView()
  rand(size).Tensor[T]

# Indexing
# -----------------------------------------------------------------------
import complex
import cppstl/std_complex

func item*[T](self: Tensor[T]): T =
  item(convertRawTensor(self), typedesc[T])
  ## Extract the scalar from a 0-dimensional tensor

func item*(self: Tensor[Complex32]): Complex32 =
  item(convertRawTensor(self), typedesc[Complex32]).toCppComplex().toComplex()

func item*(self: Tensor[Complex64]): Complex64 =
  item(convertRawTensor(self), typedesc[Complex64]).toCppComplex().toComplex()

# Unsure what those corresponds to in Python
# func `[]`*[T](self: Tensor, index: Scalar): Tensor {.inline.}
# func `[]`*[T](self: Tensor, index: Tensor): Tensor {.inline.}
# func `[]`*[T](self: Tensor, index: int64): Tensor {.inline.}

func index*[T](self: Tensor[T], args: varargs): Tensor[T] {.inline.} =
  ## Tensor indexing. It is recommended
  ## to Nimify this in a high-level wrapper.
  ## `tensor.index(indexers)`
  index(convertRawTensor(self), args).Tensor[T]
# We can't use the construct `#.index_put_({@}, #)`
# so hardcode sizes,
# 6d seems reasonable, that would be a batch of 3D videos (videoID/batchID, Time, Color Channel, Height, Width, Depth)
# If you need more you likely aren't indexing individual values.

func index_put*[T](self: var Tensor[T], idx: varargs[int|int64], val: T or Tensor[T]) {.inline.} =
  ## Tensor mutation at index. It is recommended
  index_put(convertRawTensor(self), idx, val)

# Fancy Indexing
# -----------------------------------------------------------------------
func index_select*[T](self: Tensor[T], axis: int64, indices: Tensor[T]): Tensor[T] {.inline.} =
  index_select(convertRawTensor(self), axis, indices).Tensor[T]

func masked_select*[T](self: Tensor[T], mask: Tensor[T]): Tensor[T] {.inline.} =
  masked_select(convertRawTensor(self), convertRawTensor(mask)).Tensor[T]

# PyTorch exposes in-place `index_fill_` and `masked_fill_`
# and out-of-place `index_fill` and `masked_fill`
# that does in-place + clone
# we only exposes the in-place version.

func index_fill_mut*[T](self: var Tensor[T], mask: Tensor[T], value: T or Tensor[T]) {.inline.} =
  index_fill_mut(convertRawTensor(self), convertRawTensor(mask), value)

func masked_fill_mut*[T](self: var Tensor[T], mask: Tensor[T], value: T or Tensor[T]) {.inline.} =
  masked_fill_mut(convertRawTensor(self), convertRawTensor(mask), value)

# Shapeshifting
# -----------------------------------------------------------------------

func reshape*[T](self: Tensor[T], shape: openArray[int64]): Tensor[T] {.inline.} =
  let sizes = sizes.asTorchView()
  reshape(convertRawTensor(self), sizes).Tensor[T]

func view*[T](self: Tensor[T], size: openArray[int64]): Tensor[T] {.inline.} =
  let size = size.asTorchView()
  reshape(convertRawTensor(self), size).Tensor[T]

# Automatic Differentiation
# -----------------------------------------------------------------------

func backward*[T](self: var Tensor[T]) {.inline.} =
  backward(convertRawTensor(self))

# Operators
# -----------------------------------------------------------------------
## TODO FINISH TODO
#
# func `not`*[T](self: Tensor[T]): Tensor[T] {.inline.}
# func `-`*[T](self: Tensor[T]): Tensor[T] {.inline.}
#
# func `+`*[T](self: Tensor[T], b: Tensor[T]): Tensor[T] {.inline.}
# func `-`*[T](self: Tensor[T], b: Tensor[T]): Tensor[T] {.inline.}
# func `*[T]`*[T](self: Tensor[T], b: Tensor[T]): Tensor[T] {.inline.}
#
# func `*[T]`*[T](a: cfloat or cdouble, b: Tensor[T]): Tensor[T] {.inline.}
# func `*[T]`*[T](self: Tensor[T], b: cfloat or cdouble): Tensor[T] {.inline.}
#
# func `+=`*[T](self: var Tensor[T], b: Tensor[T]) {.inline.}
# func `+=`*[T](self: var Tensor[T], s: Scalar) {.inline.}
# func `-=`*[T](self: var Tensor[T], b: Tensor[T]) {.inline.}
# func `-=`*[T](self: var Tensor[T], s: Scalar) {.inline.}
# func `*[T]=`*[T](self: var Tensor[T], b: Tensor[T]) {.inline.}
# func `*[T]=`*[T](self: var Tensor[T], s: Scalar) {.inline.}
# func `/=`*[T](self: var Tensor[T], b: Tensor[T]) {.inline.}
# func `/=`*[T](self: var Tensor[T], s: Scalar) {.inline.}
#
# func `and`*[T](self: Tensor[T], b: Tensor[T]): Tensor[T] {.inline.}
#   ## bitwise `and`.
# func `or`*[T](self: Tensor[T], b: Tensor[T]): Tensor[T] {.inline.}
#   ## bitwise `or`.
# func `xor`*[T](self: Tensor[T], b: Tensor[T]): Tensor[T] {.inline.}
#   ## bitwise `xor`.
#
# func bitand_mut*[T](self: var Tensor[T], s: Tensor[T]) {.inline.}
#   ## In-place bitwise `and`.
# func bitor_mut*[T](self: var Tensor[T], s: Tensor[T]) {.inline.}
#   ## In-place bitwise `or`.
# func bitxor_mut*[T](self: var Tensor[T], s: Tensor[T]) {.inline.}
#   ## In-place bitwise `xor`.
#
# func eq*[T](a, b: Tensor[T]): Tensor[T] {.inline.}
#   ## Equality of each tensor values
# func equal*[T](a, b: Tensor[T]): bool {.inline.}
# template `==`*[T](a, b: Tensor[T]): bool =
#   a.equal(b)
#
# # Functions.h
# # -----------------------------------------------------------------------
#
# func toType*[T](self: Tensor[T], dtype: ScalarKind): Tensor[T] {.inline.}
# func toSparse*[T](self: Tensor[T]): Tensor[T] {.inline.}
# func toSparse*[T](self: Tensor[T], sparseDim: int64): Tensor[T] {.inline.}
#
# func eye*[T](n: int64): Tensor[T] {.inline.}
# func eye*[T](n: int64, options: TensorOptions): Tensor[T] {.inline.}
# func eye*[T](n: int64, scalarKind: ScalarKind): Tensor[T] {.inline.}
# func eye*[T](n: int64, device: DeviceKind): Tensor[T] {.inline.}
#
# func zeros*[T](dim: int64): Tensor[T] {.inline.}
# func zeros*[T](dim: IntArrayRef): Tensor[T] {.inline.}
# func zeros*[T](dim: IntArrayRef, options: TensorOptions): Tensor[T] {.inline.}
# func zeros*[T](dim: IntArrayRef, scalarKind: ScalarKind): Tensor[T] {.inline.}
# func zeros*[T](dim: IntArrayRef, device: DeviceKind): Tensor[T] {.inline.}
#
# func linspace*[T](start, stop: Scalar, steps: int64, options: TensorOptions): Tensor[T] {.inline.}
# func linspace*[T](start, stop: Scalar, steps: int64, options: ScalarKind): Tensor[T] {.inline.}
# func linspace*[T](start, stop: Scalar, steps: int64, options: DeviceKind): Tensor[T] {.inline.}
# func linspace*[T](start, stop: Scalar, steps: int64, options: Device): Tensor[T] {.inline.}
# func linspace*[T](start, stop: Scalar, steps: int64): Tensor[T] {.inline.}
# func linspace*[T](start, stop: Scalar): Tensor[T] {.inline.}
#
# func logspace*[T](start, stop: Scalar, steps, base: int64, options: TensorOptions): Tensor[T] {.inline.}
# func logspace*[T](start, stop: Scalar, steps, base: int64, options: ScalarKind): Tensor[T] {.inline.}
# func logspace*[T](start, stop: Scalar, steps, base: int64, options: DeviceKind) {.inline.}
# func logspace*[T](start, stop: Scalar, steps, base: int64, options: Device): Tensor[T] {.inline.}
# func logspace*[T](start, stop: Scalar, steps, base: int64): Tensor[T] {.inline.}
# func logspace*[T](start, stop: Scalar, steps: int64): Tensor[T] {.inline.}
# func logspace*[T](start, stop: Scalar): Tensor[T] {.inline.}
#
# func arange*[T](stop: Scalar, options: TensorOptions): Tensor[T] {.inline.}
# func arange*[T](stop: Scalar, options: ScalarKind): Tensor[T] {.inline.}
# func arange*[T](stop: Scalar, options: DeviceKind): Tensor[T] {.inline.}
# func arange*[T](stop: Scalar, options: Device): Tensor[T] {.inline.}
# func arange*[T](stop: Scalar): Tensor[T] {.inline.}
#
# func arange*[T](start, stop: Scalar, options: TensorOptions): Tensor[T] {.inline.}
# func arange*[T](start, stop: Scalar, options: ScalarKind): Tensor[T] {.inline.}
# func arange*[T](start, stop: Scalar, options: DeviceKind): Tensor[T] {.inline.}
# func arange*[T](start, stop: Scalar, options: Device): Tensor[T] {.inline.}
# func arange*[T](start, stop: Scalar): Tensor[T] {.inline.}
#
# func arange*[T](start, stop, step: Scalar, options: TensorOptions): Tensor[T] {.inline.}
# func arange*[T](start, stop, step: Scalar, options: ScalarKind): Tensor[T] {.inline.}
# func arange*[T](start, stop, step: Scalar, options: DeviceKind): Tensor[T] {.inline.}
# func arange*[T](start, stop, step: Scalar, options: Device): Tensor[T] {.inline.}
# func arange*[T](start, stop, step: Scalar): Tensor[T] {.inline.}
#
# # Operations
# # -----------------------------------------------------------------------
# func add*[T](self: Tensor[T], other: Tensor[T], alpha: Scalar = 1): Tensor[T] {.inline.}
# func add*[T](self: Tensor[T], other: Scalar, alpha: Scalar = 1): Tensor[T] {.inline.}
# func addmv*[T](self: Tensor[T], mat: Tensor[T], vec: Tensor[T], beta: Scalar = 1, alpha: Scalar = 1): Tensor[T] {.inline.}
# func addmm*[T](t, mat1, mat2: Tensor[T], beta: Scalar = 1, alpha: Scalar = 1): Tensor[T] {.inline.}
# func mm*[T](t, other: Tensor[T]): Tensor[T] {.inline.}
# func matmul*[T](t, other: Tensor[T]): Tensor[T] {.inline.}
# func bmm*[T](t, other: Tensor[T]): Tensor[T] {.inline.}
#
# func luSolve*[T](t, data, pivots: Tensor[T]): Tensor[T] {.inline.}
#
# func qr*[T](self: Tensor[T], some: bool = true): CppTuple2[Tensor[T], Tensor[T]] {.inline.}
#   ## Returns a tuple:
#   ## - Q of shape (∗,m,k)
#   ## - R of shape (∗,k,n)
#   ## with k=min(m,n) if some is true otherwise k=m
#   ##
#   ## The QR decomposition is batched over dimension(s) *[T]
#   ## t = QR
#
# # addr?
# func all*[T](self: Tensor[T], axis: int64): Tensor[T] {.inline.}
# func all*[T](self: Tensor[T], axis: int64, keepdim: bool): Tensor[T] {.inline.}
# func allClose*[T](t, other: Tensor[T], rtol: float64 = 1e-5, abstol: float64 = 1e-8, equalNan: bool = false): bool {.inline.}
# func any*[T](self: Tensor[T], axis: int64): Tensor[T] {.inline.}
# func any*[T](self: Tensor[T], axis: int64, keepdim: bool): Tensor[T] {.inline.}
# func argmax*[T](self: Tensor[T]): Tensor[T] {.inline.}
# func argmax*[T](self: Tensor[T], axis: int64, keepdim: bool = false): Tensor[T] {.inline.}
# func argmin*[T](self: Tensor[T]): Tensor[T] {.inline.}
# func argmin*[T](self: Tensor[T], axis: int64, keepdim: bool = false): Tensor[T] {.inline.}
#
# # aggregate
# # -----------------------------------------------------------------------
#
# # sum needs wrapper procs/templates to allow for using nim arrays and single axis.
# func sum*[T](self: Tensor[T]): Tensor[T] {.inline.}
# func sum*[T](self: Tensor[T], dtype: ScalarKind): Tensor[T] {.inline.}
# func sum*[T](self: Tensor[T], axis: int64, keepdim: bool = false): Tensor[T] {.inline.}
# func sum*[T](self: Tensor[T], axis: int64, keepdim: bool = false, dtype: ScalarKind): Tensor[T] {.inline.}
# func sum*[T](self: Tensor[T], axis: IntArrayRef, keepdim: bool = false): Tensor[T] {.inline.}
# func sum*[T](self: Tensor[T], axis: IntArrayRef, keepdim: bool = false, dtype: ScalarKind): Tensor[T] {.inline.}
#
# # mean as well
# func mean*[T](self: Tensor[T]): Tensor[T] {.inline.}
# func mean*[T](self: Tensor[T], dtype: ScalarKind): Tensor[T] {.inline.}
# func mean*[T](self: Tensor[T], axis: int64, keepdim: bool = false): Tensor[T] {.inline.}
# func mean*[T](self: Tensor[T], axis: int64, keepdim: bool = false, dtype: ScalarKind): Tensor[T] {.inline.}
# func mean*[T](self: Tensor[T], axis: IntArrayRef, keepdim: bool = false): Tensor[T] {.inline.}
# func mean*[T](self: Tensor[T], axis: IntArrayRef, keepdim: bool = false, dtype: ScalarKind): Tensor[T] {.inline.}
#
# # median requires std::tuple
#
# func prod*[T](self: Tensor[T]): Tensor[T] {.inline.}
# func prod*[T](self: Tensor[T], dtype: ScalarKind): Tensor[T] {.inline.}
# func prod*[T](self: Tensor[T], axis: int64, keepdim: bool = false): Tensor[T] {.inline.}
# func prod*[T](self: Tensor[T], axis: int64, keepdim: bool = false, dtype: ScalarKind): Tensor[T] {.inline.}
#
# func min*[T](self: Tensor[T]): Tensor[T] {.inline.}
# func min*[T](self: Tensor[T], axis: int64, keepdim: bool = false): CppTuple2[Tensor[T], Tensor[T]] {.inline.}
#   ## Returns a tuple (values, indices) of type (TensorT, TensorInt64)
#   ## of the minimum values and their index in the specified axis
#
# func max*[T](self: Tensor[T]): Tensor[T] {.inline.}
# func max*[T](self: Tensor[T], axis: int64, keepdim: bool = false): CppTuple2[Tensor[T], Tensor[T]] {.inline.}
#   ## Returns a tuple (values, indices) of type (TensorT, TensorInt64)
#   ## of the maximum values and their index in the specified axis
#
# func variance*[T](self: Tensor[T], unbiased: bool = true): Tensor[T] {.inline.}
# func variance*[T](self: Tensor[T], axis: int64, unbiased: bool = true, keepdim: bool = false): Tensor[T] {.inline.}
# func variance*[T](self: Tensor[T], axis: IntArrayRef, unbiased: bool = true, keepdim: bool = false): Tensor[T] {.inline.}
#
# func stddev*[T](self: Tensor[T], unbiased: bool = true): Tensor[T] {.inline.}
# func stddev*[T](self: Tensor[T], axis: int64, unbiased: bool = true, keepdim: bool = false): Tensor[T] {.inline.}
# func stddev*[T](self: Tensor[T], axis: IntArrayRef, unbiased: bool = true, keepdim: bool = false): Tensor[T] {.inline.}
#
# # algorithms:
# # -----------------------------------------------------------------------
#
# func sort*[T](self: Tensor[T], axis: int64 = -1, descending: bool = false): CppTuple2[Tensor[T], Tensor[T]] {.inline.}
#   ## Sorts the elements of the input tensor along a given dimension in ascending order by value.
#   ## If dim is not given, the last dimension of the input is chosen (dim=-1).
#   ## Returns (values, originalIndices) or type (TensorT, TensorInt64)
#   ## where originalIndices is the original index of each values (before sorting)
# func argsort*[T](self: Tensor[T], axis: int64 = -1, descending: bool = false): Tensor[T] {.inline.}
#
# # math
# # -----------------------------------------------------------------------
# func abs*[T](self: Tensor[T]): Tensor[T] {.inline.}
# func absolute*[T](self: Tensor[T]): Tensor[T] {.inline.}
# func angle*[T](self: Tensor[T]): Tensor[T] {.inline.}
# func sgn*[T](self: Tensor[T]): Tensor[T] {.inline.}
# func conj*[T](self: Tensor[T]): Tensor[T] {.inline.}
# func acos*[T](self: Tensor[T]): Tensor[T] {.inline.}
# func arccos*[T](self: Tensor[T]): Tensor[T] {.inline.}
# func acosh*[T](self: Tensor[T]): Tensor[T] {.inline.}
# func arccosh*[T](self: Tensor[T]): Tensor[T] {.inline.}
# func asinh*[T](self: Tensor[T]): Tensor[T] {.inline.}
# func arcsinh*[T](self: Tensor[T]): Tensor[T] {.inline.}
# func atanh*[T](self: Tensor[T]): Tensor[T] {.inline.}
# func arctanh*[T](self: Tensor[T]): Tensor[T] {.inline.}
# func asin*[T](self: Tensor[T]): Tensor[T] {.inline.}
# func arcsin*[T](self: Tensor[T]): Tensor[T] {.inline.}
# func atan*[T](self: Tensor[T]): Tensor[T] {.inline.}
# func arctan*[T](self: Tensor[T]): Tensor[T] {.inline.}
# func cos*[T](self: Tensor[T]): Tensor[T] {.inline.}
# func sin*[T](self: Tensor[T]): Tensor[T] {.inline.}
# func tan*[T](self: Tensor[T]): Tensor[T] {.inline.}
# func exp*[T](self: Tensor[T]): Tensor[T] {.inline.}
# func exp2*[T](self: Tensor[T]): Tensor[T] {.inline.}
# func erf*[T](self: Tensor[T]): Tensor[T] {.inline.}
# func erfc*[T](self: Tensor[T]): Tensor[T] {.inline.}
# func reciprocal*[T](self: Tensor[T]): Tensor[T] {.inline.}
# func neg*[T](self: Tensor[T]): Tensor[T] {.inline.}
# func clamp*[T](self: Tensor[T], min, max: Scalar): Tensor[T] {.inline.}
# func clampMin*[T](self: Tensor[T], min: Scalar): Tensor[T] {.inline.}
# func clampMax*[T](self: Tensor[T], max: Scalar): Tensor[T] {.inline.}
#
# func dot*[T](self: Tensor[T], other: Tensor[T]): Tensor[T] {.inline.}
#
# func squeeze*[T](self: Tensor[T]): Tensor[T] {.inline.}
# func squeeze*[T](self: Tensor[T], axis: int64): Tensor[T] {.inline.}
# func unsqueeze*[T](self: Tensor[T], axis: int64): Tensor[T] {.inline.}
#
# # FFT
# # -----------------------------------------------------------------------
# func fftshift*[T](self: Tensor[T]): Tensor[T] {.inline.}
# func fftshift*[T](self: Tensor[T], dim: IntArrayRef): Tensor[T] {.inline.}
# func ifftshift*[T](self: Tensor[T]): Tensor[T] {.inline.}
# func ifftshift*[T](self: Tensor[T], dim: IntArrayRef): Tensor[T] {.inline.}
#
# let defaultNorm: CppString = initCppString("backward")
#
# func fft*[T](self: Tensor[T], n: int64, dim: int64 = -1, norm: CppString = defaultNorm): Tensor[T] {.inline.}
# ## Compute the 1-D Fourier transform
# ## ``n`` represent Signal length. If given, the input will either be zero-padded or trimmed to this length before computing the FFT.
# ## ``norm`` can be :
# ##    *[T] "forward" - normalize by 1/n
# ##    *[T] "backward" - no normalization
# ##    *[T] "ortho" - normalize by 1/sqrt(n)
# func fft*[T](self: Tensor[T]): Tensor[T] {.inline.}
# ## Compute the 1-D Fourier transform
#
# func ifft*[T](self: Tensor[T], n: int64, dim: int64 = -1, norm: CppString = defaultNorm): Tensor[T] {.inline.}
# ## Compute the 1-D Fourier transform
# ## ``n`` represent Signal length. If given, the input will either be zero-padded or trimmed to this length before computing the FFT.
# ## ``norm`` can be :
# ##   *[T] "forward" - no normalization
# ##   *[T] "backward" - normalization by 1/n
# ##   *[T] "ortho" - normalization by 1/sqrt(n)
# func ifft*[T](self: Tensor[T]): Tensor[T] {.inline.}
# ## Compute the 1-D Fourier transform
#
# func fft2*[T](self: Tensor[T], s: IntArrayRef, dim: IntArrayRef, norm: CppString = defaultNorm): Tensor[T] {.inline.}
# ## Compute the 2-D Fourier transform
# ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
# ## ``norm`` can be :
# ##    *[T] "forward" - normalize by 1/n
# ##    *[T] "backward" - no normalization
# ##    *[T] "ortho" - normalize by 1/sqrt(n)
# ## With n the logical FFT size: ``n = prod(s)``.
# func fft2*[T](self: Tensor[T], s: IntArrayRef): Tensor[T] {.inline.}
# ## Compute the 2-D Fourier transform
# ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
# func fft2*[T](self: Tensor[T]): Tensor[T] {.inline.}
# ## Compute the 2-D Fourier transform
#
# func ifft2*[T](self: Tensor[T], s: IntArrayRef, dim: IntArrayRef, norm: CppString = defaultNorm): Tensor[T] {.inline.}
# ## Compute the 2-D Inverse Fourier transform
# ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
# ## ``norm`` can be :
# ##   *[T] "forward" - no normalization
# ##   *[T] "backward" - normalization by 1/n
# ##   *[T] "ortho" - normalization by 1/sqrt(n)
# ## With n the logical FFT size: ``n = prod(s)``.
# func ifft2*[T](self: Tensor[T], s: IntArrayRef): Tensor[T] {.inline.}
# ## Compute the 2-D Inverse Fourier transform
# ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
# func ifft2*[T](self: Tensor[T]): Tensor[T] {.inline.}
# ## Compute the 2-D Inverse Fourier transform
#
# func fftn*[T](self: Tensor[T], s: IntArrayRef, dim: IntArrayRef, norm: CppString = defaultNorm): Tensor[T] {.inline.}
# ## Compute the N-D Fourier transform
# ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
# ## ``norm`` can be :
# ##    *[T] "forward" normalize by 1/n
# ##    *[T] "backward" - no normalization
# ##    *[T] "ortho" normalize by 1/sqrt(n)
# ## With n the logical FFT size: ``n = prod(s)``.
# func fftn*[T](self: Tensor[T], s: IntArrayRef): Tensor[T] {.inline.}
# ## Compute the N-D Fourier transform
# ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
# func fftn*[T](self: Tensor[T]): Tensor[T] {.inline.}
# ## Compute the N-D Fourier transform
#
# func ifftn*[T](self: Tensor[T], s: IntArrayRef, dim: IntArrayRef, norm: CppString = defaultNorm): Tensor[T] {.inline.}
# ## Compute the N-D Inverse Fourier transform
# ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
# ## ``norm`` can be :
# ##   *[T] "forward" - no normalization
# ##   *[T] "backward" - normalization by 1/n
# ##   *[T] "ortho" - normalization by 1/sqrt(n)
# ## With n the logical FFT size: ``n = prod(s)``.
# func ifftn*[T](self: Tensor[T], s: IntArrayRef): Tensor[T] {.inline.}
# ## Compute the N-D Inverse Fourier transform
# ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
# func ifftn*[T](self: Tensor[T]): Tensor[T] {.inline.}
# ## Compute the N-D Inverse Fourier transform
#
# func rfft*[T](self: Tensor[T], n: int64, dim: int64 = -1, norm: CppString = defaultNorm): Tensor[T] {.inline.}
# ## Computes the one dimensional Fourier transform of real-valued input.
# func rfft*[T](self: Tensor[T]): Tensor[T] {.inline.}
# ## Computes the one dimensional Fourier transform of real-valued input.
#
# func irfft*[T](self: Tensor[T], n: int64, dim: int64 = -1, norm: CppString = defaultNorm): Tensor[T] {.inline.}
# ## Computes the one dimensional Fourier transform of real-valued input.
# func irfft*[T](self: Tensor[T]): Tensor[T] {.inline.}
# ## Computes the one dimensional Fourier transform of real-valued input.
#
# func rfft2*[T](self: Tensor[T], s: IntArrayRef, dim: IntArrayRef, norm: CppString = defaultNorm): Tensor[T] {.inline.}
# ## Compute the N-D Fourier transform
# ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
# ## ``norm`` can be :
# ##    *[T] "forward" - normalize by 1/n
# ##    *[T] "backward" - no normalization
# ##    *[T] "ortho" - normalize by 1/sqrt(n)
# ## With n the logical FFT size: ``n = prod(s)``.
# func rfft2*[T](self: Tensor[T], s: IntArrayRef): Tensor[T] {.inline.}
# ## Compute the N-D Fourier transform
# ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
# func rfft2*[T](self: Tensor[T]): Tensor[T] {.inline.}
# ## Compute the N-D Fourier transform
#
# func irfft2*[T](self: Tensor[T], s: IntArrayRef, dim: IntArrayRef, norm: CppString = defaultNorm): Tensor[T] {.inline.}
# ## Compute the N-D Inverse Fourier transform
# ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
# ## ``norm`` can be :
# ##   *[T] "forward" - no normalization
# ##   *[T] "backward" - normalization by 1/n
# ##   *[T] "ortho" - normalization by 1/sqrt(n)
# ## With n the logical FFT size: ``n = prod(s)``.
# func irfft2*[T](self: Tensor[T], s: IntArrayRef): Tensor[T] {.inline.}
# ## Compute the N-D Inverse Fourier transform
# ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
# func irfft2*[T](self: Tensor[T]): Tensor[T] {.inline.}
# ## Compute the N-D Inverse Fourier transform
#
#
# func rfftn*[T](self: Tensor[T], s: IntArrayRef, dim: IntArrayRef, norm: CppString = defaultNorm): Tensor[T] {.inline.}
# ## Compute the N-D Fourier transform
# ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
# ## ``norm`` can be :
# ##    *[T] "forward" - normalize by 1/n
# ##    *[T] "backward" - no normalization
# ##    *[T] "ortho" - normalize by 1/sqrt(n)
# ## With n the logical FFT size: ``n = prod(s)``.
# func rfftn*[T](self: Tensor[T], s: IntArrayRef): Tensor[T] {.inline.}
# ## Compute the N-D Fourier transform
# ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
# func rfftn*[T](self: Tensor[T]): Tensor[T] {.inline.}
# ## Compute the N-D Fourier transform
#
# func irfftn*[T](self: Tensor[T], s: IntArrayRef, dim: IntArrayRef, norm: CppString = defaultNorm): Tensor[T] {.inline.}
# ## Compute the N-D Inverse Fourier transform
# ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
# ## ``norm`` can be :
# ##   *[T] "forward" - no normalization
# ##   *[T] "backward" - normalization by 1/n
# ##   *[T] "ortho" - normalization by 1/sqrt(n)
# ## With n the logical FFT size: ``n = prod(s)``.
# func irfftn*[T](self: Tensor[T], s: IntArrayRef): Tensor[T] {.inline.}
# ## Compute the N-D Inverse Fourier transform
# ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
# func irfftn*[T](self: Tensor[T]): Tensor[T] {.inline.}
# ## Compute the N-D Inverse Fourier transform
#
# func hfft*[T](self: Tensor[T], n: int64, dim: int64 = -1, norm: CppString = defaultNorm): Tensor[T] {.inline.}
# ## Computes the 1 dimensional FFT of a onesided Hermitian signal.
# func hfft*[T](self: Tensor[T]): Tensor[T] {.inline.}
# ## Computes the 1 dimensional FFT of a onesided Hermitian signal.
# func ihfft*[T](self: Tensor[T], n: int64, dim: int64 = -1, norm: CppString = defaultNorm): Tensor[T] {.inline.}
# ## Computes the inverse FFT of a real-valued Fourier domain signal.
# func ihfft*[T](self: Tensor[T]): Tensor[T] {.inline.}
# ## Computes the inverse FFT of a real-valued Fourier domain signal.
# #func convolution*[T](self: Tensor, weight: Tensor, bias: Tensor, stride, padding, dilation: int64, transposed: bool, outputPadding: int64, groups: int64): Tensor {.inline.}
