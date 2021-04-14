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
{.push inline.}
proc convertRawTensor*[T](t: Tensor[T]): RawTensor {.noinit.} =
  t.raw

proc convertTensor*[T](t: RawTensor): Tensor[T] {.noinit.} =
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
  convertTensor[T](
    cpu(convertRawTensor(self))
  )

func cuda*[T](self: Tensor[T]): Tensor[T] {.noinit.} =
  convertTensor[T](
    cuda(convertRawTensor(self))
  )

func hip*[T](self: Tensor[T]): Tensor[T] {.noinit.} =
  convertTensor[T](
    hip(convertRawTensor(self))
  )

func vulkan*[T](self: Tensor[T]): Tensor[T] {.noinit.} =
  convertTensor[T](
  vulkan(convertRawTensor(self))
  )

func to*[T](self: Tensor[T], device: DeviceKind): Tensor[T] {.noinit.} =
  convertTensor[T](
    to(convertRawTensor(self), device)
  )

func to*[T](self: Tensor[T], device: Device): Tensor[T] {.noinit.} =
  convertTensor[T](
    to(convertRawTensor(self), device)
  )

# dtype
# -----------------------------------------------------------------------
func to*[T](self: Tensor[T], dtype: typedesc[SomeTorchType]): Tensor[dtype] {.noinit.} =
  # Use typedesc -> ScalarKind converter
  convertTensor[dtype](
    rawtensors.to(convertRawTensor(self), dtype)
  )

func scalarType*[T](self: Tensor[T]): typedesc[T] =
  rawtensors.scalarType(convertRawTensor(self))

# Constructors
# -----------------------------------------------------------------------
# DeviceType and ScalarType are auto-convertible to TensorOptions

func from_blob*[T](data: pointer, sizes: openArray[int64], options: TensorOptions|DeviceKind): Tensor[T] {.noinit.} =
  let sizes = sizes.asTorchView
  convertTensor[T](
    rawtensors.from_blob(data, sizes, options)
  )

func from_blob*[T](data: pointer, sizes: openArray[int64]): Tensor[T] {.noinit.} =
  let sizes = sizes.asTorchView
  convertTensor[T](
    rawtensors.from_blob(data, sizes, T)
  )

func from_blob*[T](data: pointer, sizes: int64, options: TensorOptions|DeviceKind): Tensor[T] {.noinit.} =
  convertTensor[T](
    rawtensors.from_blob(data, sizes, options)
  )

func from_blob*[T](data: pointer, sizes: int64): Tensor[T] {.noinit.} =
  convertTensor[T](
    rawtensors.from_blob(data, sizes, T)
  )

func from_blob*[T](data: pointer, sizes, strides: openArray[int64], options: TensorOptions|DeviceKind): Tensor[T] {.noinit.} =
  let
    sizes = sizes.asTorchView
    strides = strides.asTorchView
  convertTensor[T](
    rawtensors.from_blob(data, sizes, strides, options)
  )

func from_blob*[T](data: pointer, sizes, strides: openArray[int64]): Tensor[T] {.noinit.} =
  let
    sizes = sizes.asTorchView
    strides = strides.asTorchView
  convertTensor[T](
    rawtensors.from_blob(data, sizes, strides, T)
  )

func empty*[T](size: openArray[int64], options: TensorOptions|DeviceKind): Tensor[T] {.noinit.} =
  ## Create an uninitialized tensor of shape `size`
  ## The tensor data must be filled manually
  ##
  ## The output tensor will be row major (C contiguous)
  let size = size.asTorchView()
  convertTensor[T](
    rawtensors.empty(size, options)
  )

func empty*[T](size: openArray[int64]): Tensor[T] {.noinit.} =
  let size = size.asTorchView()
  convertTensor[T](
    rawtensors.empty(size, T)
  )

func clone*[T](self: Tensor[T]): Tensor[T] {.noinit.} =
  convertTensor[T](
    rawtensors.clone(convertRawTensor(self))
  )

# Random sampling
# -----------------------------------------------------------------------

func random_mut*[T](self: var Tensor[T], start, stopEx: int64) =
  random_mut(convertRawTensor(self), start, stopEx)

func randint*[T](start, stopEx: int64, args: varargs): Tensor[T] {.noinit.} =
  convertTensor[T](
    rawtensors.randint(start, stopEx, args)
  )

func randint*[T](start, stopEx: int64, size: openArray[int64]): Tensor[T] {.noinit.} =
  let size = size.asTorchView()
  convertTensor[T](
    rawtensors.randint(start, stopEx, size)
  )

func rand_like*[T](self: Tensor[T], options: TensorOptions|DeviceKind|Device): Tensor[T] {.noinit.} =
  convertTensor[T](
    rawtensors.rand_like(convertRawTensor(self), options)
  )

func rand_like*[T](self: Tensor[T]): Tensor[T] {.noinit.} =
  convertTensor[T](
    rawtensors.rand_like(convertRawTensor(self), T)
  )

func rand*[T](size: openArray[int64]): Tensor[T] {.noinit.} =
  let size = size.asTorchView()
  convertTensor[T](
    rawtensors.rand(size)
  )

# Indexing
# -----------------------------------------------------------------------
import complex
import cppstl/std_complex

func item*[T](self: Tensor[T]): T =
  ## Extract the scalar from a 0-dimensional tensor
  result = item(convertRawTensor(self), T)

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
  convertTensor[T](
    index(convertRawTensor(self), args)
  )
# We can't use the construct `#.index_put_({@}, #)`
# so hardcode sizes,
# 6d seems reasonable, that would be a batch of 3D videos (videoID/batchID, Time, Color Channel, Height, Width, Depth)
# If you need more you likely aren't indexing individual values.

func index_put*[T](self: var Tensor[T], idx: varargs[int|int64], val: T or Tensor[T]) =
  ## Tensor mutation at index. It is recommended
  convertTensor[T](
    index_put(convertRawTensor(self), idx, val)
  )

# Fancy Indexing
# -----------------------------------------------------------------------
func index_select*[T](self: Tensor[T], axis: int64, indices: Tensor[T]): Tensor[T] {.noinit.} =
  convertTensor[T](
    index_select(convertRawTensor(self), axis, indices)
  )

func masked_select*[T](self: Tensor[T], mask: Tensor[T]): Tensor[T] {.noinit.} =
  convertTensor[T](
    masked_select(convertRawTensor(self), convertRawTensor(mask))
  )

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
  convertTensor[T](
    reshape(convertRawTensor(self), sizes)
  )

func view*[T](self: Tensor[T], size: openArray[int64]): Tensor[T] {.noinit.} =
  let size = size.asTorchView()
  convertTensor[T](
    reshape(convertRawTensor(self), size)
  )

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

func `*`*[T](a: SomeNumber, b: Tensor[T]): Tensor[T] =
  convertTensor[T](a.cdouble * convertRawTensor(b))

func `*`*[T](a: Tensor[T], b: SomeNumber): Tensor[T] =
  convertTensor[T](convertRawTensor(a) * b.cdouble)

{.pop.}

func `+=`*[T](self: var Tensor[T], b: Tensor[T]) =
  convertRawTensor(self) += convertRawTensor(b)

func `+=`*[T](self: var Tensor[T], s: T) =
  convertRawTensor(self) += s

func `-=`*[T](self: var Tensor[T], b: Tensor[T]) =
  convertRawTensor(self) -= convertRawTensor(b)
func `-=`*[T](self: var Tensor[T], s: T) =
  convertRawTensor(self) -= s

func `*=`*[T](self: var Tensor[T], b: Tensor[T]) =
  convertRawTensor(self) *= convertRawTensor(b)

func `*=`*[T](self: var Tensor[T], s: T) =
  convertRawTensor(self) *= s

func `/=`*[T](self: var Tensor[T], b: Tensor[T]) =
  convertRawTensor(self) /= convertRawTensor(b)
func `/=`*[T](self: var Tensor[T], s: T) =
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
  convertTensor[T](
    rawtensors.eq(convertRawTensor(a), convertRawTensor(b))
  )
{.pop.}

func equal*[T](a, b: Tensor[T]): bool =
  equal(convertRawTensor(a), convertRawTensor(b))

template `==`*[T](a, b: Tensor[T]): bool =
  a.equal(b)

# # Functions.h
# # -----------------------------------------------------------------------
{.push noinit.}
func toType*[T](self: Tensor[T], dtype: ScalarKind): Tensor[T] =
  convertTensor[T](
    rawtensors.toType(convertRawTensor(self), dtype)
  )

func toSparse*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.toSparse(convertRawTensor(self))
  )

func toSparse*[T](self: Tensor[T], sparseDim: int64): Tensor[T] =
  convertTensor[T](
    rawtensors.toSparse(convertRawTensor(self), sparseDim)
  )

func eye*[T](n: int64): Tensor[T] =
  convertTensor[T](
    rawtensors.eye(n, T)
  )

func eye*[T](n: int64, options: DeviceKind|TensorOptions): Tensor[T] =
  convertTensor[T](
    rawtensors.eye(n, options)
  )

func zeros*[T](dim: int64): Tensor[T] =
  convertTensor[T](
    rawtensors.zeros(dim)
  )

func zeros*[T](dims: openArray[int64]): Tensor[T] =
  let dims = dims.asTorchView()
  convertTensor[T](
    rawtensors.zeros(dims, T)
  )

func zeros*[T](dims: openArray[int64], options: DeviceKind|TensorOptions): Tensor[T] =
  let dims = dims.asTorchView()
  convertTensor[T](
    rawtensors.zeros(dims, options)
  )

func linspace*[T](start, stop: T, steps: int64, options: TensorOptions): Tensor[T] =
  convertTensor[T](
    rawtensors.linspace(start, stop, steps, options)
  )

func linspace*[T](start, stop: T, steps: int64, options: DeviceKind): Tensor[T] =
  convertTensor[T](
    rawtensors.linspace(start, stop, steps, options)
  )

func linspace*[T](start, stop: T, steps: int64, options: Device): Tensor[T] =
  convertTensor[T](
    rawtensors.linspace(start, stop, steps, options)
  )

func linspace*[T](start, stop: T, steps: int64): Tensor[T] =
  convertTensor[T](
    rawtensors.linspace(start, stop, steps, T)
  )

func linspace*[T](start, stop: T): Tensor[T] =
  convertTensor[T](
    rawtensors.linspace(start, stop)
  )

func logspace*[T](start, stop: T, steps: int64, options: TensorOptions): Tensor[T] =
  convertTensor[T](
    rawtensors.logspace(start, stop, steps, options)
  )

func logspace*[T](start, stop: T, steps: int64, options: DeviceKind): Tensor[T] =
  convertTensor[T](
    rawtensors.logspace(start, stop, steps, options)
  )

func logspace*[T](start, stop: T, steps: int64, options: Device): Tensor[T] =
  convertTensor[T](
    rawtensors.logspace(start, stop, steps, options)
  )

func logspace*[T](start, stop: T, steps: int64): Tensor[T] =
  convertTensor[T](
    rawtensors.logspace(start, stop, steps, T)
  )

func logspace*[T](start, stop: T): Tensor[T] =
  convertTensor[T](
    rawtensors.logspace(start, stop)
  )

func arange*[T](stop: T, options: TensorOptions): Tensor[T] =
  convertTensor[T](
    rawtensors.arange(stop, options)
  )

func arange*[T](stop: T, options: DeviceKind): Tensor[T] =
  convertTensor[T](
    rawtensors.arange(stop, options)
  )

func arange*[T](stop: T, options: Device): Tensor[T] =
  convertTensor[T](
    rawtensors.arange(stop, options)
  )

func arange*[T](stop: T): Tensor[T] =
  convertTensor[T](
    rawtensors.arange(stop, T)
  )

func arange*[T](start, stop: T, options: TensorOptions): Tensor[T] =
  convertTensor[T](
    rawtensors.arange(start, stop, options)
  )
func arange*[T](start, stop: T, options: DeviceKind): Tensor[T] =
  convertTensor[T](
    rawtensors.arange(start, stop, options)
  )
func arange*[T](start, stop: T, options: Device): Tensor[T] =
  convertTensor[T](
    rawtensors.arange(start, stop, options)
  )
func arange*[T](start, stop: T): Tensor[T] =
  convertTensor[T](
    rawtensors.arange(start, stop, T)
  )

func arange*[T](start, stop, step: T, options: TensorOptions): Tensor[T] =
  convertTensor[T](
    rawtensors.arange(start, stop, step, options)
  )
func arange*[T](start, stop, step: T, options: DeviceKind): Tensor[T] =
  convertTensor[T](
    rawtensors.arange(start, stop, step, options)
  )
func arange*[T](start, stop, step: T, options: Device): Tensor[T] =
  convertTensor[T](
    rawtensors.arange(start, stop, step, options)
  )
func arange*[T](start, stop, step: T): Tensor[T] =
  convertTensor[T](
    rawtensors.arange(start, stop, step, T)
  )

# Operations
# -----------------------------------------------------------------------
func add*[T](self: Tensor[T], other: Tensor[T], alpha: Scalar = 1): Tensor[T] =
  convertTensor[T](
    rawtensors.add(self, other, alpha)
  )
func add*[T](self: Tensor[T], other: Scalar, alpha: Scalar = 1): Tensor[T] =
  convertTensor[T](
    rawtensors.add(self, other, alpha)
  )

func addmv*[T](self: Tensor[T], mat: Tensor[T], vec: Tensor[T], beta: Scalar = 1, alpha: Scalar = 1): Tensor[T] =
  convertTensor[T](
    rawtensors.addmv(self, mat, vec, beta, alpha)
  )

func addmm*[T](t, mat1, mat2: Tensor[T], beta: Scalar = 1, alpha: Scalar = 1): Tensor[T] =
  convertTensor[T](
    rawtensors.addmm(t, mat1, mat2, beta, alpha)
  )

func mm*[T](t, other: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.mm(t, other)
  )

func matmul*[T](t, other: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.matmul(t, other)
  )

func bmm*[T](t, other: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.bmm(t, other)
  )

func luSolve*[T](t, data, pivots: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.luSolve(t, data, pivots)
  )

func qr*[T](self: Tensor[T], some: bool = true): tuple[q: Tensor[T], r: Tensor[T]] =
  ## Returns a tuple:
  ## - Q of shape (∗,m,k)
  ## - R of shape (∗,k,n)
  ## with k=min(m,n) if some is true otherwise k=m
  ##
  ## The QR decomposition is batched over dimension(s) *[T]
  ## t = QR
  let cppTupRes = rawtensors.qr(self, some)
  result.q = convertTensor[T](cppTupRes.get(0))
  result.r = convertTensor[T](cppTupRes.get(1))

# addr?
func all*[T](self: Tensor[T], axis: int64): Tensor[T] =
  convertTensor[T](
    rawtensors.all(convertRawTensor(self), axis)
  )
func all*[T](self: Tensor[T], axis: int64, keepdim: bool): Tensor[T] =
  convertTensor[T](
  rawtensors.all(convertRawTensor(self), axis, keepdim)
  )

func any*[T](self: Tensor[T], axis: int64): Tensor[T] =
  convertTensor[T](
    rawtensors.any(convertRawTensor(self), axis)
  )
func any*[T](self: Tensor[T], axis: int64, keepdim: bool): Tensor[T] =
  convertTensor[T](
    rawtensors.any(convertRawTensor(self), axis, keepdim)
  )

func argmax*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.argmax(convertRawTensor(self))
  )
func argmax*[T](self: Tensor[T], axis: int64, keepdim: bool = false): Tensor[T] =
  convertTensor[T](
    rawtensors.argmax(convertRawTensor(self), axis, keepdim)
  )

func argmin*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.argmin(convertRawTensor(self))
  )
func argmin*[T](self: Tensor[T], axis: int64, keepdim: bool = false): Tensor[T] =
  convertTensor[T](
    rawtensors.argmin(convertRawTensor(self), axis, keepdim)
  )
{.pop.}

func allClose*[T](t, other: Tensor[T], rtol: float64 = 1e-5, abstol: float64 = 1e-8, equalNan: bool = false): bool =
  allClose(convertRawTensor(t), convertRawTensor(other), rtol, abstol, equalNan)

# # aggregate
# # -----------------------------------------------------------------------
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

func min*[T](self: Tensor[T], axis: int64, keepdim: bool = false): tuple[values: Tensor[T], indices: Tensor[int64]] =
  ## Returns a tuple (values, indices) of type (TensorT, TensorInt64)
  ## of the minimum values and their index in the specified axis
  let cppMinTuple = rawtensors.min(convertRawTensor(self), axis, keepdim)
  result.values = convertTensor[T](cppMinTuple.get(0))
  result.indices = convertTensor[int64](cppMinTuple.get(1))

func max*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.max(convertRawTensor(self))
  )


func max*[T](self: Tensor[T], axis: int64, keepdim: bool = false): tuple[values: Tensor[T], indices: Tensor[int64]] =
  ## Returns a tuple (values, indices) of type (TensorT, TensorInt64)
  ## of the maximum values and their index in the specified axis
  let cppMaxTuple = rawtensors.max(convertRawTensor(self), axis, keepdim)
  result.values = convertTensor[T](cppMaxTuple.get(0))
  result.indices = convertTensor[int64](cppMaxTuple.get(1))


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


# # algorithms:
# # -----------------------------------------------------------------------
#
func sort*[T](self: Tensor[T], axis: int64 = -1, descending: bool = false): tuple[values: Tensor[T], originalIndices: Tensor[int64]] =
  ## Sorts the elements of the input tensor along a given dimension in ascending order by value.
  ## If dim is not given, the last dimension of the input is chosen (dim=-1).
  ## Returns (values, originalIndices) or type (TensorT, TensorInt64)
  ## where originalIndices is the original index of each values (before sorting)
  let cppSortTuple = rawtensors.sort(convertRawTensor(self), axis, descending)
  result.values = convertTensor[T](cppSortTuple.get(0))
  result.originalIndices = convertTensor[int64](cppSortTuple.get(1))

func argsort*[T](self: Tensor[T], axis: int64 = -1, descending: bool = false): Tensor[T] =
  convertTensor[T](
    rawtensors.argsort(convertRawTensor(self), axis, descending)
  )
#
# # math
# # -----------------------------------------------------------------------
func abs*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.abs(convertRawTensor(self))
  )

func absolute*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.absolute(convertRawTensor(self))
  )

func angle*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.angle(convertRawTensor(self))
  )

func sgn*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.sgn(convertRawTensor(self))
  )

func conj*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.conj(convertRawTensor(self))
  )

func acos*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.acos(convertRawTensor(self))
  )

func arccos*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.arccos(convertRawTensor(self))
  )

func acosh*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.acosh(convertRawTensor(self))
  )

func arccosh*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.arccosh(convertRawTensor(self))
  )

func asinh*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.asinh(convertRawTensor(self))
  )

func arcsinh*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.arcsinh(convertRawTensor(self))
  )

func atanh*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.atanh(convertRawTensor(self))
  )

func arctanh*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.arctanh(convertRawTensor(self))
  )

func asin*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.asin(convertRawTensor(self))
  )

func arcsin*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.arcsin(convertRawTensor(self))
  )

func atan*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.atan(convertRawTensor(self))
  )

func arctan*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.arctan(convertRawTensor(self))
  )

func cos*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.cos(convertRawTensor(self))
  )

func sin*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.sin(convertRawTensor(self))
  )

func tan*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.tan(convertRawTensor(self))
  )

func exp*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.exp(convertRawTensor(self))
  )

func exp2*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.exp2(convertRawTensor(self))
  )

func erf*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.erf(convertRawTensor(self))
  )

func erfc*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.erfc(convertRawTensor(self))
  )

func reciprocal*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.reciprocal(convertRawTensor(self))
  )

func neg*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.neg(convertRawTensor(self))
  )

func clamp*[T](self: Tensor[T], min, max: Scalar): Tensor[T] =
  convertTensor[T](
    rawtensors.clamp(convertRawTensor(self), min, max)
  )

func clampMin*[T](self: Tensor[T], min: Scalar): Tensor[T] =
  convertTensor[T](
    rawtensors.clampMin(convertRawTensor(self), min)
  )

func clampMax*[T](self: Tensor[T], max: Scalar): Tensor[T] =
  convertTensor[T](
    rawtensors.clampMax(convertRawTensor(self), max)
  )

func dot*[T](self: Tensor[T], other: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.dot(convertRawTensor(self), convertRawTensor(other))
  )

func squeeze*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.squeeze(convertRawTensor(self))
  )

func squeeze*[T](self: Tensor[T], axis: int64): Tensor[T] =
  convertTensor[T](
    rawtensors.squeeze(convertRawTensor(self), axis)
  )

func unsqueeze*[T](self: Tensor[T], axis: int64): Tensor[T] =
  convertTensor[T](
    rawtensors.unsqueeze(convertRawTensor(self), axis)
  )

# FFT
# -----------------------------------------------------------------------
func fftshift*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.fftshift(convertRawTensor(self))
  )

func fftshift*[T](self: Tensor[T], dims: openArray[int64]): Tensor[T] =
  let dims = dims.asTorchView()
  convertTensor[T](
    rawtensors.ifftshift(convertRawTensor(self), dims)
  )

func ifftshift*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.ifftshift(convertRawTensor(self))
  )

func ifftshift*[T](self: Tensor[T], dims: openArray[int64]): Tensor[T] =
  let dims = dims.asTorchView()
  convertTensor[T](
    rawtensors.ifftshift(convertRawTensor(self), dims)
  )

let defaultNorm: CppString = initCppString("backward")

func fft*[T](self: Tensor[T], n: int64, dim: int64 = -1, norm: CppString = defaultNorm): Tensor[T] =
  ## Compute the 1-D Fourier transform
  ## ``n`` represent Signal length. If given, the input will either be zero-padded or trimmed to this length before computing the FFT.
  ## ``norm`` can be :
  ##    *[T] "forward" - normalize by 1/n
  ##    *[T] "backward" - no normalization
  ##    *[T] "ortho" - normalize by 1/sqrt(n)
  convertTensor[T](
    rawtensors.fft(convertRawTensor(self), n, dim, norm)
  )

func fft*[T](self: Tensor[T]): Tensor[T] =
  ## Compute the 1-D Fourier transform
  convertTensor[T](
    rawtensors.fft(convertRawTensor(self))
  )

func ifft*[T](self: Tensor[T], n: int64, dim: int64 = -1, norm: CppString = defaultNorm): Tensor[T] =
  ## Compute the 1-D Fourier transform
  ## ``n`` represent Signal length. If given, the input will either be zero-padded or trimmed to this length before computing the FFT.
  ## ``norm`` can be :
  ##   *[T] "forward" - no normalization
  ##   *[T] "backward" - normalization by 1/n
  ##   *[T] "ortho" - normalization by 1/sqrt(n)
  convertTensor[T](
    rawtensors.ifft(convertRawTensor(self), n, dim, norm)
  )

func ifft*[T](self: Tensor[T]): Tensor[T] =
  ## Compute the 1-D Fourier transform
  convertTensor[T](
    rawtensors.ifft(convertRawTensor(self))
  )

func fft2*[T](self: Tensor[T], s: openArray[int64], dims: openArray[int64], norm: CppString = defaultNorm): Tensor[T] =
  ## Compute the 2-D Fourier transform
  ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
  ## ``norm`` can be :
  ##    *[T] "forward" - normalize by 1/n
  ##    *[T] "backward" - no normalization
  ##    *[T] "ortho" - normalize by 1/sqrt(n)
  ## With n the logical FFT size: ``n = prod(s)``.
  let s = s.asTorchView()
  let dims = dims.asTorchView()
  convertTensor[T](
    rawtensors.fft2(convertRawTensor(self), s, dims, norm)
  )

func fft2*[T](self: Tensor[T], s: openArray[int64]): Tensor[T] =
  ## Compute the 2-D Fourier transform
  ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
  let s = s.asTorchView()
  convertTensor[T](
    rawtensors.fft2(convertRawTensor(self), s)
  )

func fft2*[T](self: Tensor[T]): Tensor[T] =
  ## Compute the 2-D Fourier transform
  convertTensor[T](
    rawtensors.fft2(convertRawTensor(self))
  )

func ifft2*[T](self: Tensor[T], s: openArray[int64], dims: openArray[int64], norm: CppString = defaultNorm): Tensor[T] =
  ## Compute the 2-D Inverse Fourier transform
  ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
  ## ``norm`` can be :
  ##   *[T] "forward" - no normalization
  ##   *[T] "backward" - normalization by 1/n
  ##   *[T] "ortho" - normalization by 1/sqrt(n)
  ## With n the logical FFT size: ``n = prod(s)``.
  let s = s.asTorchView()
  let dims = dims.asTorchView()
  convertTensor[T](
    rawtensors.ifft2(convertRawTensor(self), s, dims, norm)
  )

func ifft2*[T](self: Tensor[T], s: openArray[int64]): Tensor[T] =
  ## Compute the 2-D Inverse Fourier transform
  ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
  let s = s.asTorchView()
  convertTensor[T](
    rawtensors.ifft2(convertRawTensor(self), s)
  )

func ifft2*[T](self: Tensor[T]): Tensor[T] =
  ## Compute the 2-D Inverse Fourier transform
  convertTensor[T](
    rawtensors.ifft2(convertRawTensor(self))
  )

func fftn*[T](self: Tensor[T], s: openArray[int64], dims: openArray[int64], norm: CppString = defaultNorm): Tensor[T] =
  ## Compute the N-D Fourier transform
  ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
  ## ``norm`` can be :
  ##    *[T] "forward" normalize by 1/n
  ##    *[T] "backward" - no normalization
  ##    *[T] "ortho" normalize by 1/sqrt(n)
  ## With n the logical FFT size: ``n = prod(s)``.
  let s = s.asTorchView()
  let dims = dims.asTorchView()
  convertTensor[T](
    rawtensors.fftn(convertRawTensor(self), s, dims)
  )

func fftn*[T](self: Tensor[T], s: openArray[int64]): Tensor[T] =
  ## Compute the N-D Fourier transform
  ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
  let s = s.asTorchView()
  convertTensor[T](
    rawtensors.fftn(convertRawTensor(self), s)
  )

func fftn*[T](self: Tensor[T]): Tensor[T] =
  ## Compute the N-D Fourier transform
  convertTensor[T](
    rawtensors.fftn(convertRawTensor(self))
  )

func ifftn*[T](self: Tensor[T], s: openArray[int64], dims: openArray[int64], norm: CppString = defaultNorm): Tensor[T] =
  ## Compute the N-D Inverse Fourier transform
  ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
  ## ``norm`` can be :
  ##   *[T] "forward" - no normalization
  ##   *[T] "backward" - normalization by 1/n
  ##   *[T] "ortho" - normalization by 1/sqrt(n)
  ## With n the logical FFT size: ``n = prod(s)``.
  let s = s.asTorchView()
  let dims = dims.asTorchView()
  convertTensor[T](
    rawtensors.fftn(convertRawTensor(self), s, dims)
  )

func ifftn*[T](self: Tensor[T], s: openArray[int64]): Tensor[T] =
  ## Compute the N-D Inverse Fourier transform
  ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
  let s = s.asTorchView()
  convertTensor[T](
    rawtensors.fftn(convertRawTensor(self), s)
  )

func ifftn*[T](self: Tensor[T]): Tensor[T] =
  ## Compute the N-D Inverse Fourier transform
  convertTensor[T](
    rawtensors.ifftn(convertRawTensor(self))
  )

# RFFT
# -----------------------------------------------------------------------

func rfft*[T](self: Tensor[T], n: int64, dim: int64 = -1, norm: CppString = defaultNorm): Tensor[T] =
  ## Compute the 1-D Fourier transform of real-valued input
  ## ``n`` represent Signal length. If given, the input will either be zero-padded or trimmed to this length before computing the rfft.
  ## ``norm`` can be :
  ##    *[T] "forward" - normalize by 1/n
  ##    *[T] "backward" - no normalization
  ##    *[T] "ortho" - normalize by 1/sqrt(n)
  convertTensor[T](
    rawtensors.rfft(convertRawTensor(self), n, dim, norm)
  )

func rfft*[T](self: Tensor[T]): Tensor[T] =
  ## Compute the 1-D Fourier transform of real-valued input
  convertTensor[T](
    rawtensors.rfft(convertRawTensor(self))
  )

func irfft*[T](self: Tensor[T], n: int64, dim: int64 = -1, norm: CppString = defaultNorm): Tensor[T] =
  ## Compute the 1-D Fourier transform of real-valued input
  ## ``n`` represent Signal length. If given, the input will either be zero-padded or trimmed to this length before computing the rfft.
  ## ``norm`` can be :
  ##   *[T] "forward" - no normalization
  ##   *[T] "backward" - normalization by 1/n
  ##   *[T] "ortho" - normalization by 1/sqrt(n)
  convertTensor[T](
    rawtensors.irfft(convertRawTensor(self), n, dim, norm)
  )

func irfft*[T](self: Tensor[T]): Tensor[T] =
  ## Compute the 1-D Fourier transform of real-valued input
  convertTensor[T](
    rawtensors.irfft(convertRawTensor(self))
  )

func rfft2*[T](self: Tensor[T], s: openArray[int64], dims: openArray[int64], norm: CppString = defaultNorm): Tensor[T] =
  ## Compute the 2-D Fourier transform of real-valued input
  ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the rfft.
  ## ``norm`` can be :
  ##    *[T] "forward" - normalize by 1/n
  ##    *[T] "backward" - no normalization
  ##    *[T] "ortho" - normalize by 1/sqrt(n)
  ## With n the logical rfft size: ``n = prod(s)``.
  let s = s.asTorchView()
  let dims = dims.asTorchView()
  convertTensor[T](
    rawtensors.rfft2(convertRawTensor(self), s, dims, norm)
  )

func rfft2*[T](self: Tensor[T], s: openArray[int64]): Tensor[T] =
  ## Compute the 2-D Fourier transform of real-valued input
  ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the rfft.
  let s = s.asTorchView()
  convertTensor[T](
    rawtensors.rfft2(convertRawTensor(self), s)
  )

func rfft2*[T](self: Tensor[T]): Tensor[T] =
  ## Compute the 2-D Fourier transform of real-valued input
  convertTensor[T](
    rawtensors.rfft2(convertRawTensor(self))
  )

func irfft2*[T](self: Tensor[T], s: openArray[int64], dims: openArray[int64], norm: CppString = defaultNorm): Tensor[T] =
  ## Compute the 2-D Inverse Fourier transform of real-valued input
  ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the rfft.
  ## ``norm`` can be :
  ##   *[T] "forward" - no normalization
  ##   *[T] "backward" - normalization by 1/n
  ##   *[T] "ortho" - normalization by 1/sqrt(n)
  ## With n the logical rfft size: ``n = prod(s)``.
  let s = s.asTorchView()
  let dims = dims.asTorchView()
  convertTensor[T](
    rawtensors.irfft2(convertRawTensor(self), s, dims, norm)
  )

func irfft2*[T](self: Tensor[T], s: openArray[int64]): Tensor[T] =
  ## Compute the 2-D Inverse Fourier transform of real-valued input
  ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the rfft.
  let s = s.asTorchView()
  convertTensor[T](
    rawtensors.irfft2(convertRawTensor(self), s)
  )

func irfft2*[T](self: Tensor[T]): Tensor[T] =
  ## Compute the 2-D Inverse Fourier transform of real-valued input
  convertTensor[T](
    rawtensors.irfft2(convertRawTensor(self))
  )

func rfftn*[T](self: Tensor[T], s: openArray[int64], dims: openArray[int64], norm: CppString = defaultNorm): Tensor[T] =
  ## Compute the N-D Fourier transform of real-valued input
  ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the rfft.
  ## ``norm`` can be :
  ##    *[T] "forward" normalize by 1/n
  ##    *[T] "backward" - no normalization
  ##    *[T] "ortho" normalize by 1/sqrt(n)
  ## With n the logical rfft size: ``n = prod(s)``.
  let s = s.asTorchView()
  let dims = dims.asTorchView()
  convertTensor[T](
    rawtensors.rfftn(convertRawTensor(self), s, dims)
  )

func rfftn*[T](self: Tensor[T], s: openArray[int64]): Tensor[T] =
  ## Compute the N-D Fourier transform of real-valued input
  ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the rfft.
  let s = s.asTorchView()
  convertTensor[T](
    rawtensors.rfftn(convertRawTensor(self), s)
  )

func rfftn*[T](self: Tensor[T]): Tensor[T] =
  ## Compute the N-D Fourier transform of real-valued input
  convertTensor[T](
    rawtensors.rfftn(convertRawTensor(self))
  )

func irfftn*[T](self: Tensor[T], s: openArray[int64], dims: openArray[int64], norm: CppString = defaultNorm): Tensor[T] =
  ## Compute the N-D Inverse Fourier transform of real-valued input
  ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the rfft.
  ## ``norm`` can be :
  ##   *[T] "forward" - no normalization
  ##   *[T] "backward" - normalization by 1/n
  ##   *[T] "ortho" - normalization by 1/sqrt(n)
  ## With n the logical rfft size: ``n = prod(s)``.
  let s = s.asTorchView()
  let dims = dims.asTorchView()
  convertTensor[T](
    rawtensors.rfftn(convertRawTensor(self), s, dims)
  )

func irfftn*[T](self: Tensor[T], s: openArray[int64]): Tensor[T] =
  ## Compute the N-D Inverse Fourier transform of real-valued input
  ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the rfft.
  let s = s.asTorchView()
  convertTensor[T](
    rawtensors.rfftn(convertRawTensor(self), s)
  )

func irfftn*[T](self: Tensor[T]): Tensor[T] =
  ## Compute the N-D Inverse Fourier transform
  convertTensor[T](
    rawtensors.irfftn(convertRawTensor(self))
  )

func hfft*[T](self: Tensor[T], n: int64, dim: int64 = -1, norm: CppString = defaultNorm): Tensor[T] =
  ## Computes the 1 dimensional FFT of a onesided Hermitian signal.
  convertTensor[T](
    rawtensors.hfft(convertRawTensor(self), n, dim, norm)
  )

func hfft*[T](self: Tensor[T]): Tensor[T] =
  ## Computes the 1 dimensional FFT of a onesided Hermitian signal.
  convertTensor[T](
    rawtensors.hfft(convertRawTensor(self))
  )

func ihfft*[T](self: Tensor[T], n: int64, dim: int64 = -1, norm: CppString = defaultNorm): Tensor[T] =
  ## Computes the inverse FFT of a real-valued Fourier domain signal.
  convertTensor[T](
    rawtensors.ihfft(convertRawTensor(self), n, dim, norm)
  )

func ihfft*[T](self: Tensor[T]): Tensor[T] =
  ## Computes the inverse FFT of a real-valued Fourier domain signal.
  convertTensor[T](
    rawtensors.ihfft(convertRawTensor(self))
  )
{.pop.}

# #func convolution*[T](self: Tensor, weight: Tensor, bias: Tensor, stride, padding, dilation: int64, transposed: bool, outputPadding: int64, groups: int64): Tensor
