import raw/bindings/[rawtensors, c10]
import raw/cpp/[std_cpp]
import raw/sugar/[interop, indexing]
import std/[complex, macros]

export SomeTorchType

{.experimental: "views".} # TODO

type
  Tensor*[T] {.requiresinit.} = object
    ## calling zeroMem on Tensor - which is called as default initialization - will set the refcount to 0 of the internal intrusive_ptr<TensorImpl> and destroy the RawTensor causin a segùentation fault
    ## It is imperative to either declare tensor object a ``noinit``, initialize specifically Tensor using ``initTensor``.
    ## In addition, all proc that return a Tensor object used as constructor must be declared as ``noinit``.
    raw*: RawTensor

proc initTensor[T](): Tensor[T] {.constructor, noinit.} =
  {.emit: "/* */".}

# proc `=copy`*[T](dest: var Tensor[T], src: Tensor[T]) =
#   dest.raw = src.raw.clone()
#
# proc `=sink`*[T](dest: var Tensor[T], src: Tensor[T]) =
#   `=destroy`(dest)
#   wasMoved(dest)
#   dest.raw = src.raw

proc convertRawTensor*[T](t: Tensor[T]): RawTensor {.noinit, inline.} =
  t.raw

proc convertRawTensor*[T](t: var Tensor[T]): var RawTensor {.noinit, inline.} =
  t.raw

proc convertTensor*[T](t: RawTensor): Tensor[T] {.noinit, inline.} =
  # assign(result.raw, t)
  result.raw = t

proc convertTensor*[T](t: var RawTensor): var Tensor[T] {.noinit, inline.} =
  # assign(result.raw, t)
  result.raw = t

# Strings & Debugging
# -----------------------------------------------------------------------

proc print*[T](self: Tensor[T]) {.sideeffect.} =
  print(convertRawTensor(self))

# Metadata
# -----------------------------------------------------------------------

func dim*[T](self: Tensor[T]): int64 =
  ## Number of dimensions
  dim(convertRawTensor(self))

func reset*[T](self: var Tensor[T]) =
  reset(convertRawTensor(self))

func is_same*[T](self, other: Tensor[T]): bool =
  ## Reference equality
  ## Do the tensors use the same memory.
  is_same(convertRawTensor(self), convertRawTensor(other))

func sizes*[T](self: Tensor[T]): IntArrayRef =
  ## This is Arraymancer and Numpy "shape"
  sizes(convertRawTensor(self))

func shape*[T](self: Tensor[T]): seq[int64] =
  ## This is Arraymancer and Numpy "shape"
  result = asNimView(sizes(convertRawTensor(self)))

func strides*[T](self: Tensor[T]): openArray[int64] =
  strides(convertRawTensor(self))

func ndimension*[T](self: Tensor[T]): int64 =
  ## This is Arraymancer rank
  ndimension(convertRawTensor(self))

func nbytes*[T](self: Tensor[T]): uint =
  ## Bytes-size of the Tensor
  nbytes(convertRawTensor(self))

func numel*[T](self: Tensor[T]): int64 =
  ## This is Arraymancer and Numpy "size"
  numel(convertRawTensor(self))

func size*[T](self: Tensor[T], axis: int64): int64 =
  size(convertRawTensor(self))

func itemsize*[T](self: Tensor[T]): uint =
  itemsize(convertRawTensor(self))

func element_size*[T](self: Tensor[T]): int64 =
  element_size(convertRawTensor(self))

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
  data_ptr(convertRawTensor(self), T)

# Backend
# -----------------------------------------------------------------------

func has_storage*[T](self: Tensor[T]): bool =
  has_storage(convertRawTensor(self))

func get_device*[T](self: Tensor[T]): int64 =
  get_device(convertRawTensor(self))

func is_cuda*[T](self: Tensor[T]): bool =
  is_cuda(convertRawTensor(self))

func is_hip*[T](self: Tensor[T]): bool =
  is_hip(convertRawTensor(self))

func is_sparse*[T](self: Tensor[T]): bool =
  is_sparse(convertRawTensor(self))

func is_mkldnn*[T](self: Tensor[T]): bool =
  is_mkldnn(convertRawTensor(self))

func is_vulkan*[T](self: Tensor[T]): bool =
  is_vulkan(convertRawTensor(self))

func is_quantized*[T](self: Tensor[T]): bool =
  is_quantized(convertRawTensor(self))

func is_meta*[T](self: Tensor[T]): bool =
  is_meta(convertRawTensor(self))

func cpu*[T](self: Tensor[T]): Tensor[T] {.noinit.} =
  cpu(convertRawTensor(self))

func cuda*[T](self: Tensor[T]): Tensor[T] {.noinit.} =
  cuda(convertRawTensor(self))

func hip*[T](self: Tensor[T]): Tensor[T] {.noinit.} =
  hip(convertRawTensor(self))

func vulkan*[T](self: Tensor[T]): Tensor[T] {.noinit.} =
  vulkan(convertRawTensor(self))

func to*[T](self: Tensor[T], device: DeviceKind): Tensor[T] {.noinit.} =
  to(convertRawTensor(self), device)

func to*[T](self: Tensor[T], device: Device): Tensor[T] {.noinit.} =
  to(convertRawTensor(self), device)

# dtype
# -----------------------------------------------------------------------
func to*[T](self: Tensor[T], dtype: typedesc[SomeTorchType]): Tensor[T] {.noinit.} =
  # Use typedesc -> ScalarKind converter
  to(convertRawTensor(self), dtype)

func scalarType*[T](self: Tensor[T]): typedesc[T] =
  rawtensors.scalarType(convertRawTensor(self))

# Constructors
# -----------------------------------------------------------------------
# DeviceType and ScalarType are auto-convertible to TensorOptions

func from_blob*[T](data: pointer, sizes: openArray[int64], options: TensorOptions|DeviceKind): Tensor[T] {.noinit.} =
  let sizes = sizes.asTorchView
  rawtensors.from_blob(data, sizes, options).convertTensor[T]

func from_blob*[T](data: pointer, sizes: openArray[int64]): Tensor[T] {.noinit.} =
  let sizes = sizes.asTorchView
  rawtensors.from_blob(data, sizes, T).convertTensor[T]

func from_blob*[T](data: pointer, sizes: int64, options: TensorOptions|DeviceKind): Tensor[T] {.noinit.} =
  rawtensors.from_blob(data, sizes, options).convertTensor[T]

func from_blob*[T](data: pointer, sizes: int64): Tensor[T] {.noinit.} =
  rawtensors.from_blob(data, sizes, T).convertTensor[T]

func from_blob*[T](data: pointer, sizes, strides: openArray[int64], options: TensorOptions|DeviceKind): Tensor[T] {.noinit.} =
  let
    sizes = sizes.asTorchView
    strides = strides.asTorchView
  rawtensors.from_blob(data, sizes, strides, options).convertTensor[T]

func from_blob*[T](data: pointer, sizes, strides: openArray[int64]): Tensor[T] {.noinit.} =
  let
    sizes = sizes.asTorchView
    strides = strides.asTorchView
  rawtensors.from_blob(data, sizes, strides, T).convertTensor[T]

func empty*[T](size: IntArrayRef, options: TensorOptions|DeviceKind): Tensor[T] {.noinit.} =
  ## Create an uninitialized tensor of shape `size`
  ## The tensor data must be filled manually
  ##
  ## The output tensor will be row major (C contiguous)
  # let size = size.asTorchView
  rawtensors.empty(size, options).convertTensor[T]

func empty*[T](size: IntArrayRef): Tensor[T] {.noinit.} =
  # let size = size.asTorchView
  rawtensors.empty(size, T).convertTensor[T]

func clone*[T](self: Tensor[T]): Tensor[T] {.noinit.} =
  rawtensors.clone(convertRawTensor(self)).convertTensor[T]

# Random sampling
# -----------------------------------------------------------------------

func random_mut*[T](self: var Tensor[T], start, stopEx: int64) =
  random_mut(convertRawTensor(self), start, stopEx)

func randint*[T](start, stopEx: int64, args: varargs): Tensor[T] {.noinit.} =
  rawtensors.randint(start, stopEx, args).convertTensor[T]

func randint*[T](start, stopEx: int64, size: openArray[int64]): Tensor[T] {.noinit.} =
  let size = size.asTorchView()
  rawtensors.randint(start, stopEx, size).convertTensor[T]

func rand_like*[T](self: Tensor[T], options: TensorOptions|DeviceKind|Device): Tensor[T] {.noinit.} =
  rawtensors.rand_like(convertRawTensor(self), options).convertTensor[T]

func rand_like*[T](self: Tensor[T]): Tensor[T] {.noinit.} =
  rawtensors.rand_like(convertRawTensor(self), T).convertTensor[T]

func rand*[T](size: openArray[int64]): Tensor[T] {.noinit.} =
  let size = size.asTorchView()
  rawtensors.rand(size).convertTensor[T]

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
# func `[]`*[T](self: Tensor, index: Scalar): Tensor
# func `[]`*[T](self: Tensor, index: Tensor): Tensor
# func `[]`*[T](self: Tensor, index: int64): Tensor

func index*[T](self: Tensor[T], args: varargs): Tensor[T] {.noinit.} =
  ## Tensor indexing. It is recommended
  ## to Nimify this in a high-level wrapper.
  ## `tensor.index(indexers)`
  index(convertRawTensor(self), args).convertTensor[T]
# We can't use the construct `#.index_put_({@}, #)`
# so hardcode sizes,
# 6d seems reasonable, that would be a batch of 3D videos (videoID/batchID, Time, Color Channel, Height, Width, Depth)
# If you need more you likely aren't indexing individual values.

func index_put*[T](self: var Tensor[T], idx: varargs[int|int64], val: T or Tensor[T]) =
  ## Tensor mutation at index. It is recommended
  index_put(convertRawTensor(self), idx, val)

# Fancy Indexing
# -----------------------------------------------------------------------
func index_select*[T](self: Tensor[T], axis: int64, indices: Tensor[T]): Tensor[T] {.noinit.} =
  index_select(convertRawTensor(self), axis, indices).convertTensor[T]

func masked_select*[T](self: Tensor[T], mask: Tensor[T]): Tensor[T] {.noinit.} =
  masked_select(convertRawTensor(self), convertRawTensor(mask)).convertTensor[T]

# PyTorch exposes in-place `index_fill_` and `masked_fill_`
# and out-of-place `index_fill` and `masked_fill`
# that does in-place + clone
# we only exposes the in-place version.

func index_fill_mut*[T](self: var Tensor[T], mask: Tensor[T], value: T or Tensor[T]) =
  index_fill_mut(convertRawTensor(self), convertRawTensor(mask), value)

func masked_fill_mut*[T](self: var Tensor[T], mask: Tensor[T], value: T or Tensor[T]) =
  masked_fill_mut(convertRawTensor(self), convertRawTensor(mask), value)

# Shapeshifting
# -----------------------------------------------------------------------

func reshape*[T](self: Tensor[T], shape: openArray[int64]): Tensor[T] {.noinit.} =
  let sizes = sizes.asTorchView()
  reshape(convertRawTensor(self), sizes).convertTensor[T]

func view*[T](self: Tensor[T], size: openArray[int64]): Tensor[T] {.noinit.} =
  let size = size.asTorchView()
  reshape(convertRawTensor(self), size).convertTensor[T]

# Automatic Differentiation
# -----------------------------------------------------------------------

func backward*[T](self: var Tensor[T]) =
  backward(convertRawTensor(self))

# Operators
# -----------------------------------------------------------------------
## TODO FINISH TODO
{.push noinit.}
func `not`*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](not convertRawTensor(self))

func `-`*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](-convertRawTensor(self))

func `+`*[T](a: Tensor[T], b: Tensor[T]): Tensor[T] =
  convertTensor[T](convertRawTensor(a) + convertRawTensor(b))

func `-`*[T](a: Tensor[T], b: Tensor[T]): Tensor[T] =
  convertTensor[T](convertRawTensor(a) - convertRawTensor(b))

func `*`*[T](a: Tensor[T], b: Tensor[T]): Tensor[T] =
  convertTensor[T](convertRawTensor(a) * convertRawTensor(b))

func `*`*[T](a: cfloat or cdouble, b: Tensor[T]): Tensor[T] =
  convertTensor[T](a * convertRawTensor(b))

func `*`*[T](a: Tensor[T], b: cfloat or cdouble): Tensor[T] =
  convertTensor[T](convertRawTensor(a) * b)

{.pop.}

func `+=`*[T](self: var Tensor[T], b: Tensor[T]) =
  convertRawTensor(self) += convertRawTensor(b)
func `+=`*[T](self: var Tensor[T], s: Scalar) =
  convertRawTensor(self) += s

func `-=`*[T](self: var Tensor[T], b: Tensor[T]) =
  convertRawTensor(self) -= convertRawTensor(b)
func `-=`*[T](self: var Tensor[T], s: Scalar) =
  convertRawTensor(self) -= s

func `*=`*[T](self: var Tensor[T], b: Tensor[T]) =
  convertRawTensor(self) *= convertRawTensor(b)

func `*=`*[T](self: var Tensor[T], s: Scalar) =
  convertRawTensor(self) *= s

func `/=`*[T](self: var Tensor[T], b: Tensor[T]) =
  convertRawTensor(self) /= convertRawTensor(b)
func `/=`*[T](self: var Tensor[T], s: Scalar) =
  convertRawTensor(self) /= s

{.push noinit.}
func `and`*[T](a: Tensor[T], b: Tensor[T]): Tensor[T] =
  ## bitwise `and`.
  convertTensor[T](convertRawTensor(a) and convertRawTensor(b))

func `or`*[T](a: Tensor[T], b: Tensor[T]): Tensor[T] =
  ## bitwise `or`.
  convertTensor[T](convertRawTensor(a) or convertRawTensor(b))

func `xor`*[T](a: Tensor[T], b: Tensor[T]): Tensor[T] =
  ## bitwise `xor`.
  convertTensor[T](convertRawTensor(a) xor convertRawTensor(b))

func bitand_mut*[T](self: var Tensor[T], s: Tensor[T]) =
  ## In-place bitwise `and`.
  rawtensors.bitand_mut(convertRawTensor(self), convertRawTensor(s))

func bitor_mut*[T](self: var Tensor[T], s: Tensor[T]) =
  ## In-place bitwise `or`.
  rawtensors.bitor_mut(convertRawTensor(self), convertRawTensor(s))

func bitxor_mut*[T](self: var Tensor[T], s: Tensor[T]) =
  ## In-place bitwise `xor`.
  rawtensors.bitxor_mut(convertRawTensor(self), convertRawTensor(s))

func eq*[T](a, b: Tensor[T]): Tensor[T] =
  ## Equality of each tensor values
  rawtensors.eq(convertRawTensor(a), convertRawTensor(b)).convertTensor[T]
{.pop.}

func equal*[T](a, b: Tensor[T]): bool =
  convertRawTensor(a) == convertRawTensor(b)

template `==`*[T](a, b: Tensor[T]): bool =
  a.equal(b)

# # Functions.h
# # -----------------------------------------------------------------------
{.push noinit.}
func toType*[T](self: Tensor[T], dtype: ScalarKind): Tensor[T] =
  rawtensors.toType(convertRawTensor(self), dtype)

func toSparse*[T](self: Tensor[T]): Tensor[T] =
  rawtensors.toSparse(convertRawTensor(self)).convertTensor[T]
func toSparse*[T](self: Tensor[T], sparseDim: int64): Tensor[T] =
  rawtensors.toSparse(convertRawTensor(self), sparseDim).convertTensor[T]

func eye*[T](n: int64): Tensor[T] =
  rawtensors.eye(n, T).convertTensor[T]
func eye*[T](n: int64, options: DeviceKind|TensorOptions): Tensor[T] =
  rawtensors.eye(n, options).convertTensor[T]

func zeros*[T](dim: int64): Tensor[T] =
  rawtensors.zeros(dim, T).convertTensor[T]
func zeros*[T](dims: IntArrayRef): Tensor[T] =
  rawtensors.zeros(dims, T).convertTensor[T]
func zeros*[T](dims: IntArrayRef, options: DeviceKind|TensorOptions): Tensor[T] =
  rawtensors.zeros(dims, options).convertTensor[T]

func linspace*[T](start, stop: Scalar, steps: int64, options: TensorOptions): Tensor[T] =
  rawtensors.linspace(start, stop, steps, options).convertTensor[T]
func linspace*[T](start, stop: Scalar, steps: int64, options: DeviceKind): Tensor[T] =
  rawtensors.linspace(start, stop, steps, options).convertTensor[T]
func linspace*[T](start, stop: Scalar, steps: int64, options: Device): Tensor[T] =
  rawtensors.linspace(start, stop, steps, options).convertTensor[T]
func linspace*[T](start, stop: Scalar, steps: int64): Tensor[T] =
  rawtensors.linspace(start, stop, steps, T).convertTensor[T]
func linspace*[T](start, stop: Scalar): Tensor[T] =
  rawtensors.linspace(start, stop).convertTensor[T]

func logspace*[T](start, stop: Scalar, steps: int64, options: TensorOptions): Tensor[T] =
  rawtensors.logspace(start, stop, steps, options).convertTensor[T]
func logspace*[T](start, stop: Scalar, steps: int64, options: DeviceKind): Tensor[T] =
  rawtensors.logspace(start, stop, steps, options).convertTensor[T]
func logspace*[T](start, stop: Scalar, steps: int64, options: Device): Tensor[T] =
  rawtensors.logspace(start, stop, steps, options).convertTensor[T]
func logspace*[T](start, stop: Scalar, steps: int64): Tensor[T] =
  rawtensors.logspace(start, stop, steps, T).convertTensor[T]
func logspace*[T](start, stop: Scalar): Tensor[T] =
  rawtensors.logspace(start, stop).convertTensor[T]

func arange*[T](stop: Scalar, options: TensorOptions): Tensor[T] =
  rawtensors.arange(stop, options).convertTensor[T]
func arange*[T](stop: Scalar, options: DeviceKind): Tensor[T] =
  rawtensors.arange(stop, options).convertTensor[T]
func arange*[T](stop: Scalar, options: Device): Tensor[T] =
  rawtensors.arange(stop, options).convertTensor[T]
func arange*[T](stop: Scalar): Tensor[T] =
  rawtensors.arange(stop, T).convertTensor[T]

func arange*[T](start, stop: Scalar, options: TensorOptions): Tensor[T] =
  rawtensors.arange(start, stop, options).convertTensor[T]
func arange*[T](start, stop: Scalar, options: DeviceKind): Tensor[T] =
  rawtensors.arange(start, stop, options).convertTensor[T]
func arange*[T](start, stop: Scalar, options: Device): Tensor[T] =
  rawtensors.arange(start, stop, options).convertTensor[T]
func arange*[T](start, stop: Scalar): Tensor[T] =
  rawtensors.arange(start, stop, T).convertTensor[T]

func arange*[T](start, stop, step: Scalar, options: TensorOptions): Tensor[T] =
  rawtensors.arange(start, stop, step, options).convertTensor[T]
func arange*[T](start, stop, step: Scalar, options: DeviceKind): Tensor[T] =
  rawtensors.arange(start, stop, step, options).convertTensor[T]
func arange*[T](start, stop, step: Scalar, options: Device): Tensor[T] =
  rawtensors.arange(start, stop, step, options).convertTensor[T]
func arange*[T](start, stop, step: Scalar): Tensor[T] =
  rawtensors.arange(start, stop, step, T).convertTensor[T]

# Operations
# -----------------------------------------------------------------------
func add*[T](self: Tensor[T], other: Tensor[T], alpha: Scalar = 1): Tensor[T] =
  rawtensors.add(self, other, alpha).convertTensor[T]
func add*[T](self: Tensor[T], other: Scalar, alpha: Scalar = 1): Tensor[T] =
  rawtensors.add(self, other, alpha).convertTensor[T]

func addmv*[T](self: Tensor[T], mat: Tensor[T], vec: Tensor[T], beta: Scalar = 1, alpha: Scalar = 1): Tensor[T] =
  rawtensors.addmv(self, mat, vec, beta, alpha).convertTensor[T]

func addmm*[T](t, mat1, mat2: Tensor[T], beta: Scalar = 1, alpha: Scalar = 1): Tensor[T] =
  rawtensors.addmm(t, mat1, mat2, beta, alpha).convertTensor[T]

func mm*[T](t, other: Tensor[T]): Tensor[T] =
  rawtensors.mm(t, other).convertTensor[T]

func matmul*[T](t, other: Tensor[T]): Tensor[T] =
  rawtensors.matmul(t, other).convertTensor[T]

func bmm*[T](t, other: Tensor[T]): Tensor[T] =
  rawtensors.bmm(t, other).convertTensor[T]

func luSolve*[T](t, data, pivots: Tensor[T]): Tensor[T] =
  rawtensors.luSolve(t, data, pivots).convertTensor[T]

func qr*[T](self: Tensor[T], some: bool = true): tuple[q: Tensor[T], r: Tensor[T]] =
  ## Returns a tuple:
  ## - Q of shape (∗,m,k)
  ## - R of shape (∗,k,n)
  ## with k=min(m,n) if some is true otherwise k=m
  ##
  ## The QR decomposition is batched over dimension(s) *[T]
  ## t = QR
  let cppTupRes = rawtensors.qr(self, some)
  result.q = cppTupRes.get(0).convertTensor[T]
  result.r = cppTupRes.get(1).convertTensor[T]

# addr?
func all*[T](self: Tensor[T], axis: int64): Tensor[T] =
  rawtensors.all(convertRawTensor(self), axis).convertTensor[T]
func all*[T](self: Tensor[T], axis: int64, keepdim: bool): Tensor[T] =
  rawtensors.all(convertRawTensor(self), axis, keepdim).convertTensor[T]

func any*[T](self: Tensor[T], axis: int64): Tensor[T] =
  rawtensors.any(convertRawTensor(self), axis).convertTensor[T]
func any*[T](self: Tensor[T], axis: int64, keepdim: bool): Tensor[T] =
  rawtensors.any(convertRawTensor(self), axis, keepdim).convertTensor[T]

func argmax*[T](self: Tensor[T]): Tensor[T] =
  rawtensors.argmax(convertRawTensor(self)).convertTensor[T]
func argmax*[T](self: Tensor[T], axis: int64, keepdim: bool = false): Tensor[T] =
  rawtensors.argmax(convertRawTensor(self), axis, keepdim).convertTensor[T]

func argmin*[T](self: Tensor[T]): Tensor[T] =
  rawtensors.argmin(convertRawTensor(self)).convertTensor[T]
func argmin*[T](self: Tensor[T], axis: int64, keepdim: bool = false): Tensor[T] =
  rawtensors.argmin(convertRawTensor(self), axis, keepdim).convertTensor[T]
{.pop.}

func allClose*[T](t, other: Tensor[T], rtol: float64 = 1e-5, abstol: float64 = 1e-8, equalNan: bool = false): bool =
  allClose(convertRawTensor(t), convertRawTensor(other), rtol, abstol, equalNan)

# # aggregate
# # -----------------------------------------------------------------------
{.push noinit.}
# sum needs wrapper procs/templates to allow for using nim arrays and single axis.
func sum*[T](self: Tensor[T]): Tensor[T]
func sum*[T](self: Tensor[T], dtype: ScalarKind): Tensor[T]
func sum*[T](self: Tensor[T], axis: int64, keepdim: bool = false): Tensor[T]
func sum*[T](self: Tensor[T], axis: int64, keepdim: bool = false, dtype: ScalarKind): Tensor[T]
func sum*[T](self: Tensor[T], axis: IntArrayRef, keepdim: bool = false): Tensor[T]
func sum*[T](self: Tensor[T], axis: IntArrayRef, keepdim: bool = false, dtype: ScalarKind): Tensor[T]

# mean as well
func mean*[T](self: Tensor[T]): Tensor[T]
func mean*[T](self: Tensor[T], dtype: ScalarKind): Tensor[T]
func mean*[T](self: Tensor[T], axis: int64, keepdim: bool = false): Tensor[T]
func mean*[T](self: Tensor[T], axis: int64, keepdim: bool = false, dtype: ScalarKind): Tensor[T]
func mean*[T](self: Tensor[T], axis: IntArrayRef, keepdim: bool = false): Tensor[T]
func mean*[T](self: Tensor[T], axis: IntArrayRef, keepdim: bool = false, dtype: ScalarKind): Tensor[T]

# median requires std::tuple

func prod*[T](self: Tensor[T]): Tensor[T]
func prod*[T](self: Tensor[T], dtype: ScalarKind): Tensor[T]
func prod*[T](self: Tensor[T], axis: int64, keepdim: bool = false): Tensor[T]
func prod*[T](self: Tensor[T], axis: int64, keepdim: bool = false, dtype: ScalarKind): Tensor[T]

func min*[T](self: Tensor[T]): Tensor[T]
func min*[T](self: Tensor[T], axis: int64, keepdim: bool = false): CppTuple2[Tensor[T], Tensor[T]]
  ## Returns a tuple (values, indices) of type (TensorT, TensorInt64)
  ## of the minimum values and their index in the specified axis

func max*[T](self: Tensor[T]): Tensor[T]
func max*[T](self: Tensor[T], axis: int64, keepdim: bool = false): CppTuple2[Tensor[T], Tensor[T]]
  ## Returns a tuple (values, indices) of type (TensorT, TensorInt64)
  ## of the maximum values and their index in the specified axis

func variance*[T](self: Tensor[T], unbiased: bool = true): Tensor[T]
func variance*[T](self: Tensor[T], axis: int64, unbiased: bool = true, keepdim: bool = false): Tensor[T]
func variance*[T](self: Tensor[T], axis: IntArrayRef, unbiased: bool = true, keepdim: bool = false): Tensor[T]

func stddev*[T](self: Tensor[T], unbiased: bool = true): Tensor[T]
func stddev*[T](self: Tensor[T], axis: int64, unbiased: bool = true, keepdim: bool = false): Tensor[T]
func stddev*[T](self: Tensor[T], axis: IntArrayRef, unbiased: bool = true, keepdim: bool = false): Tensor[T]
{.pop.}

# # algorithms:
# # -----------------------------------------------------------------------
#
func sort*[T](self: Tensor[T], axis: int64 = -1, descending: bool = false): CppTuple2[Tensor[T], Tensor[T]]
  ## Sorts the elements of the input tensor along a given dimension in ascending order by value.
  ## If dim is not given, the last dimension of the input is chosen (dim=-1).
  ## Returns (values, originalIndices) or type (TensorT, TensorInt64)
  ## where originalIndices is the original index of each values (before sorting)
func argsort*[T](self: Tensor[T], axis: int64 = -1, descending: bool = false): Tensor[T]
#
# # math
# # -----------------------------------------------------------------------
func abs*[T](self: Tensor[T]): Tensor[T]
func absolute*[T](self: Tensor[T]): Tensor[T]
func angle*[T](self: Tensor[T]): Tensor[T]
func sgn*[T](self: Tensor[T]): Tensor[T]
func conj*[T](self: Tensor[T]): Tensor[T]
func acos*[T](self: Tensor[T]): Tensor[T]
func arccos*[T](self: Tensor[T]): Tensor[T]
func acosh*[T](self: Tensor[T]): Tensor[T]
func arccosh*[T](self: Tensor[T]): Tensor[T]
func asinh*[T](self: Tensor[T]): Tensor[T]
func arcsinh*[T](self: Tensor[T]): Tensor[T]
func atanh*[T](self: Tensor[T]): Tensor[T]
func arctanh*[T](self: Tensor[T]): Tensor[T]
func asin*[T](self: Tensor[T]): Tensor[T]
func arcsin*[T](self: Tensor[T]): Tensor[T]
func atan*[T](self: Tensor[T]): Tensor[T]
func arctan*[T](self: Tensor[T]): Tensor[T]
func cos*[T](self: Tensor[T]): Tensor[T]
func sin*[T](self: Tensor[T]): Tensor[T]
func tan*[T](self: Tensor[T]): Tensor[T]
func exp*[T](self: Tensor[T]): Tensor[T]
func exp2*[T](self: Tensor[T]): Tensor[T]
func erf*[T](self: Tensor[T]): Tensor[T]
func erfc*[T](self: Tensor[T]): Tensor[T]
func reciprocal*[T](self: Tensor[T]): Tensor[T]
func neg*[T](self: Tensor[T]): Tensor[T]
func clamp*[T](self: Tensor[T], min, max: Scalar): Tensor[T]
func clampMin*[T](self: Tensor[T], min: Scalar): Tensor[T]
func clampMax*[T](self: Tensor[T], max: Scalar): Tensor[T]

func dot*[T](self: Tensor[T], other: Tensor[T]): Tensor[T]

func squeeze*[T](self: Tensor[T]): Tensor[T]
func squeeze*[T](self: Tensor[T], axis: int64): Tensor[T]
func unsqueeze*[T](self: Tensor[T], axis: int64): Tensor[T]

# # FFT
# # -----------------------------------------------------------------------
# func fftshift*[T](self: Tensor[T]): Tensor[T]
# func fftshift*[T](self: Tensor[T], dim: IntArrayRef): Tensor[T]
# func ifftshift*[T](self: Tensor[T]): Tensor[T]
# func ifftshift*[T](self: Tensor[T], dim: IntArrayRef): Tensor[T]
#
# let defaultNorm: CppString = initCppString("backward")
#
# func fft*[T](self: Tensor[T], n: int64, dim: int64 = -1, norm: CppString = defaultNorm): Tensor[T]
# ## Compute the 1-D Fourier transform
# ## ``n`` represent Signal length. If given, the input will either be zero-padded or trimmed to this length before computing the FFT.
# ## ``norm`` can be :
# ##    *[T] "forward" - normalize by 1/n
# ##    *[T] "backward" - no normalization
# ##    *[T] "ortho" - normalize by 1/sqrt(n)
# func fft*[T](self: Tensor[T]): Tensor[T]
# ## Compute the 1-D Fourier transform
#
# func ifft*[T](self: Tensor[T], n: int64, dim: int64 = -1, norm: CppString = defaultNorm): Tensor[T]
# ## Compute the 1-D Fourier transform
# ## ``n`` represent Signal length. If given, the input will either be zero-padded or trimmed to this length before computing the FFT.
# ## ``norm`` can be :
# ##   *[T] "forward" - no normalization
# ##   *[T] "backward" - normalization by 1/n
# ##   *[T] "ortho" - normalization by 1/sqrt(n)
# func ifft*[T](self: Tensor[T]): Tensor[T]
# ## Compute the 1-D Fourier transform
#
# func fft2*[T](self: Tensor[T], s: IntArrayRef, dim: IntArrayRef, norm: CppString = defaultNorm): Tensor[T]
# ## Compute the 2-D Fourier transform
# ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
# ## ``norm`` can be :
# ##    *[T] "forward" - normalize by 1/n
# ##    *[T] "backward" - no normalization
# ##    *[T] "ortho" - normalize by 1/sqrt(n)
# ## With n the logical FFT size: ``n = prod(s)``.
# func fft2*[T](self: Tensor[T], s: IntArrayRef): Tensor[T]
# ## Compute the 2-D Fourier transform
# ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
# func fft2*[T](self: Tensor[T]): Tensor[T]
# ## Compute the 2-D Fourier transform
#
# func ifft2*[T](self: Tensor[T], s: IntArrayRef, dim: IntArrayRef, norm: CppString = defaultNorm): Tensor[T]
# ## Compute the 2-D Inverse Fourier transform
# ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
# ## ``norm`` can be :
# ##   *[T] "forward" - no normalization
# ##   *[T] "backward" - normalization by 1/n
# ##   *[T] "ortho" - normalization by 1/sqrt(n)
# ## With n the logical FFT size: ``n = prod(s)``.
# func ifft2*[T](self: Tensor[T], s: IntArrayRef): Tensor[T]
# ## Compute the 2-D Inverse Fourier transform
# ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
# func ifft2*[T](self: Tensor[T]): Tensor[T]
# ## Compute the 2-D Inverse Fourier transform
#
# func fftn*[T](self: Tensor[T], s: IntArrayRef, dim: IntArrayRef, norm: CppString = defaultNorm): Tensor[T]
# ## Compute the N-D Fourier transform
# ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
# ## ``norm`` can be :
# ##    *[T] "forward" normalize by 1/n
# ##    *[T] "backward" - no normalization
# ##    *[T] "ortho" normalize by 1/sqrt(n)
# ## With n the logical FFT size: ``n = prod(s)``.
# func fftn*[T](self: Tensor[T], s: IntArrayRef): Tensor[T]
# ## Compute the N-D Fourier transform
# ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
# func fftn*[T](self: Tensor[T]): Tensor[T]
# ## Compute the N-D Fourier transform
#
# func ifftn*[T](self: Tensor[T], s: IntArrayRef, dim: IntArrayRef, norm: CppString = defaultNorm): Tensor[T]
# ## Compute the N-D Inverse Fourier transform
# ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
# ## ``norm`` can be :
# ##   *[T] "forward" - no normalization
# ##   *[T] "backward" - normalization by 1/n
# ##   *[T] "ortho" - normalization by 1/sqrt(n)
# ## With n the logical FFT size: ``n = prod(s)``.
# func ifftn*[T](self: Tensor[T], s: IntArrayRef): Tensor[T]
# ## Compute the N-D Inverse Fourier transform
# ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
# func ifftn*[T](self: Tensor[T]): Tensor[T]
# ## Compute the N-D Inverse Fourier transform
#
# func rfft*[T](self: Tensor[T], n: int64, dim: int64 = -1, norm: CppString = defaultNorm): Tensor[T]
# ## Computes the one dimensional Fourier transform of real-valued input.
# func rfft*[T](self: Tensor[T]): Tensor[T]
# ## Computes the one dimensional Fourier transform of real-valued input.
#
# func irfft*[T](self: Tensor[T], n: int64, dim: int64 = -1, norm: CppString = defaultNorm): Tensor[T]
# ## Computes the one dimensional Fourier transform of real-valued input.
# func irfft*[T](self: Tensor[T]): Tensor[T]
# ## Computes the one dimensional Fourier transform of real-valued input.
#
# func rfft2*[T](self: Tensor[T], s: IntArrayRef, dim: IntArrayRef, norm: CppString = defaultNorm): Tensor[T]
# ## Compute the N-D Fourier transform
# ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
# ## ``norm`` can be :
# ##    *[T] "forward" - normalize by 1/n
# ##    *[T] "backward" - no normalization
# ##    *[T] "ortho" - normalize by 1/sqrt(n)
# ## With n the logical FFT size: ``n = prod(s)``.
# func rfft2*[T](self: Tensor[T], s: IntArrayRef): Tensor[T]
# ## Compute the N-D Fourier transform
# ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
# func rfft2*[T](self: Tensor[T]): Tensor[T]
# ## Compute the N-D Fourier transform
#
# func irfft2*[T](self: Tensor[T], s: IntArrayRef, dim: IntArrayRef, norm: CppString = defaultNorm): Tensor[T]
# ## Compute the N-D Inverse Fourier transform
# ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
# ## ``norm`` can be :
# ##   *[T] "forward" - no normalization
# ##   *[T] "backward" - normalization by 1/n
# ##   *[T] "ortho" - normalization by 1/sqrt(n)
# ## With n the logical FFT size: ``n = prod(s)``.
# func irfft2*[T](self: Tensor[T], s: IntArrayRef): Tensor[T]
# ## Compute the N-D Inverse Fourier transform
# ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
# func irfft2*[T](self: Tensor[T]): Tensor[T]
# ## Compute the N-D Inverse Fourier transform
#
#
# func rfftn*[T](self: Tensor[T], s: IntArrayRef, dim: IntArrayRef, norm: CppString = defaultNorm): Tensor[T]
# ## Compute the N-D Fourier transform
# ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
# ## ``norm`` can be :
# ##    *[T] "forward" - normalize by 1/n
# ##    *[T] "backward" - no normalization
# ##    *[T] "ortho" - normalize by 1/sqrt(n)
# ## With n the logical FFT size: ``n = prod(s)``.
# func rfftn*[T](self: Tensor[T], s: IntArrayRef): Tensor[T]
# ## Compute the N-D Fourier transform
# ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
# func rfftn*[T](self: Tensor[T]): Tensor[T]
# ## Compute the N-D Fourier transform
#
# func irfftn*[T](self: Tensor[T], s: IntArrayRef, dim: IntArrayRef, norm: CppString = defaultNorm): Tensor[T]
# ## Compute the N-D Inverse Fourier transform
# ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
# ## ``norm`` can be :
# ##   *[T] "forward" - no normalization
# ##   *[T] "backward" - normalization by 1/n
# ##   *[T] "ortho" - normalization by 1/sqrt(n)
# ## With n the logical FFT size: ``n = prod(s)``.
# func irfftn*[T](self: Tensor[T], s: IntArrayRef): Tensor[T]
# ## Compute the N-D Inverse Fourier transform
# ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
# func irfftn*[T](self: Tensor[T]): Tensor[T]
# ## Compute the N-D Inverse Fourier transform
#
# func hfft*[T](self: Tensor[T], n: int64, dim: int64 = -1, norm: CppString = defaultNorm): Tensor[T]
# ## Computes the 1 dimensional FFT of a onesided Hermitian signal.
# func hfft*[T](self: Tensor[T]): Tensor[T]
# ## Computes the 1 dimensional FFT of a onesided Hermitian signal.
# func ihfft*[T](self: Tensor[T], n: int64, dim: int64 = -1, norm: CppString = defaultNorm): Tensor[T]
# ## Computes the inverse FFT of a real-valued Fourier domain signal.
# func ihfft*[T](self: Tensor[T]): Tensor[T]
# ## Computes the inverse FFT of a real-valued Fourier domain signal.

# #func convolution*[T](self: Tensor, weight: Tensor, bias: Tensor, stride, padding, dilation: int64, transposed: bool, outputPadding: int64, groups: int64): Tensor
