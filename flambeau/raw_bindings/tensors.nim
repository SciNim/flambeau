# Flambeau
# Copyright (c) 2020 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  # Standard library
  std/complex,
  # Internal
  ../cpp/std_cpp,
  ../libtorch,
  ./c10

# (Almost) raw bindings to PyTorch Tensors
# -----------------------------------------------------------------------
#
# This provides almost raw bindings to PyTorch tensors.
#
# "Nimification" (camelCase), ergonomic indexing and interoperability with Nim types is left to the "high-level" bindings.
# This should ease searching PyTorch and libtorch documentation,
# and make C++ tutorials easily applicable.
#
# Nonetheless some slight modifications were given to the raw bindings:
# - `&=`, `|=` and `^=` have been renamed bitand, bitor, bitxor
# - `[]` and `[]=` are not exported as index and index_put are more flexible
#   and we want to leave those symbols available for Numpy-like ergonomic indexing.
# - Nim's `index_fill_mut` and `masked_fill_mut` are mapped to the in-place
#   C++ `index_fill_` and `masked_fill_`.
#   The original out-of-place versions are doing clone+in-place mutation

# C++ interop
# -----------------------------------------------------------------------

{.push cdecl.}
{.push header: torchHeader.}

# #######################################################################
#
#                         Context
#
# #######################################################################

type Torch* = object

# Random Number Generation
# -----------------------------------------------------------------------

proc manual_seed*(_: type Torch, seed: uint64) {.sideeffect, importcpp: "torch::manual_seed(@)".}
  ## Set torch random number generator seed

# Backends
# -----------------------------------------------------------------------

proc hasCuda*(_: type Torch): bool{.sideeffect, importcpp: "torch::hasCuda()".}
  ## Returns true if libtorch was compiled with CUDA support
proc cuda_is_available*(_: type Torch): bool{.sideeffect, importcpp: "torch::cuda::is_available()".}
  ## Returns true if libtorch was compiled with CUDA support
  ## and at least one CUDA device is available
proc cudnn_is_available*(_: type Torch): bool {.sideeffect, importcpp: "torch::cuda::cudnn_is_available()".}
  ## Returns true if libtorch was compiled with CUDA and CuDNN support
  ## and at least one CUDA device is available

# #######################################################################
#
#                         Tensor Metadata
#
# #######################################################################

# Backend Device
# -----------------------------------------------------------------------
# libtorch/include/c10/core/DeviceType.h
# libtorch/include/c10/core/Device.h

type
  DeviceIndex = int16

  DeviceKind* {.importc: "c10::DeviceType",
                size: sizeof(int16).} = enum
    kCPU = 0
    kCUDA = 1
    kMKLDNN = 2
    kOpenGL = 3
    kOpenCL = 4
    kIDEEP = 5
    kHIP = 6
    kFPGA = 7
    kMSNPU = 8
    kXLA = 9
    kVulkan = 10

  Device* {.importc: "c10::Device", bycopy.} = object
    kind: DeviceKind
    index: DeviceIndex

func init*(T: type Device, kind: DeviceKind): T {.constructor, importcpp: "torch::Device(#)".}

# Datatypes
# -----------------------------------------------------------------------
# libtorch/include/torch/csrc/api/include/torch/types.h
# libtorch/include/c10/core/ScalarType.h

type
  ScalarKind* {.importc: "torch::ScalarType",
                size: sizeof(int8).} = enum
    kUint8 = 0       # kByte
    kInt8 = 1        # kChar
    kInt16 = 2       # kShort
    kInt32 = 3       # kInt
    kInt64 = 4       # kLong
    kFloat16 = 5     # kHalf
    kFloat32 = 6     # kFloat
    kFloat64 = 7     # kDouble
    kComplexF16 = 8  # kComplexHalf
    kComplexF32 = 9  # kComplexFloat
    kComplexF64 = 10 # kComplexDouble
    kBool = 11
    kQint8 = 12      # Quantized int8
    kQuint8 = 13     # Quantized uint8
    kQint32 = 14     # Quantized int32
    kBfloat16 = 15   # Brain float16


  SomeTorchType* = uint8|byte or SomeSignedInt or
                   SomeFloat or Complex[float32] or Complex[float64]
  ## Torch Tensor type mapped to Nim type

# TensorOptions
# -----------------------------------------------------------------------
# libtorch/include/c10/core/TensorOptions.h

type
  TensorOptions* {.importcpp: "torch::TensorOptions", bycopy.} = object

func init*(T: type TensorOptions): TensorOptions {.constructor, importcpp: "torch::TensorOptions".}

# Scalars
# -----------------------------------------------------------------------
# Scalars are defined in libtorch/include/c10/core/Scalar.h
# as tagged unions of double, int64, complex
# And C++ types are implicitly convertible to Scalar
#
# Hence in Nim we don't need to care about Scalar or defined converters
# (except maybe for complex)
type Scalar* = SomeNumber or bool

# TensorAccessors
# -----------------------------------------------------------------------
# libtorch/include/ATen/core/TensorAccessors.h
#
# Tensor accessors gives "medium-level" access to a Tensor raw-data
# - Compared to low-level "data_ptr" they take care of striding and shape
# - Compared to high-level functions they don't provide any parallelism.

# #######################################################################
#
#                            Tensors
#
# #######################################################################

# Tensors
# -----------------------------------------------------------------------

type
  Tensor* {.importcpp: "torch::Tensor", bycopy.} = object

# Strings & Debugging
# -----------------------------------------------------------------------

proc print*(self: Tensor) {.sideeffect, importcpp: "torch::print(@)".}

# Metadata
# -----------------------------------------------------------------------

func dim*(self: Tensor): int64 {.importcpp: "#.dim()".}
  ## Number of dimensions
func reset*(self: var Tensor) {.importcpp: "#.reset()".}
func is_same*(self, other: Tensor): bool {.importcpp: "#.is_same(#)".}
  ## Reference equality
  ## Do the tensors use the same memory.

func sizes*(self: Tensor): IntArrayRef {.importcpp: "#.sizes()".}
  ## This is Arraymancer and Numpy "shape"

func strides*(self: Tensor): IntArrayRef {.importcpp: "#.strides()".}

func ndimension*(self: Tensor): int64 {.importcpp: "#.ndimension()".}
  ## This is Arraymancer rank
func nbytes*(self: Tensor): uint {.importcpp: "#.nbytes()".}
  ## Bytes-size of the Tensor
func numel*(self: Tensor): int64 {.importcpp: "#.numel()".}
  ## This is Arraymancer and Numpy "size"

func size*(self: Tensor, axis: int64): int64 {.importcpp: "#.size(#)".}
func itemsize*(self: Tensor): uint {.importcpp: "#.itemsize()".}
func element_size*(self: Tensor): int64 {.importcpp: "#.element_size()".}

# Accessors
# -----------------------------------------------------------------------

func data_ptr*(self: Tensor, T: typedesc[SomeTorchType]): ptr UncheckedArray[T] {.importcpp: "#.data_ptr<'2>(#)".}
  ## Gives raw access to a tensor data of type T.
  ##
  ## This is a very low-level procedure. You need to take care
  ## of the tensor shape and strides yourself.
  ##
  ## It is recommended to use this only on contiguous tensors
  ## (freshly created or freshly cloned) and to avoid
  ## sliced tensors.

# Backend
# -----------------------------------------------------------------------

func has_storage*(self: Tensor): bool {.importcpp: "#.has_storage()".}
func get_device*(self: Tensor): int64 {.importcpp: "#.get_device()".}
func is_cuda*(self: Tensor): bool {.importcpp: "#.is_cuda()".}
func is_hip*(self: Tensor): bool {.importcpp: "#.is_hip()".}
func is_sparse*(self: Tensor): bool {.importcpp: "#.is_sparse()".}
func is_mkldnn*(self: Tensor): bool {.importcpp: "#.is_mkldnn()".}
func is_vulkan*(self: Tensor): bool {.importcpp: "#.is_vulkan()".}
func is_quantized*(self: Tensor): bool {.importcpp: "#.is_quantized()".}
func is_meta*(self: Tensor): bool {.importcpp: "#.is_meta()".}

func cpu*(self: Tensor): Tensor {.importcpp: "#.cpu()".}
func cuda*(self: Tensor): Tensor {.importcpp: "#.cuda()".}
func hip*(self: Tensor): Tensor {.importcpp: "#.hip()".}
func vulkan*(self: Tensor): Tensor {.importcpp: "#.vulkan()".}
func to*(self: Tensor, device: DeviceKind): Tensor {.importcpp: "#.to(#)".}
func to*(self: Tensor, device: Device): Tensor {.importcpp: "#.to(#)".}

# dtype
# -----------------------------------------------------------------------

func to*(self: Tensor, dtype: ScalarKind): Tensor {.importcpp: "#.to(#)".}
func scalarType*(self: Tensor): ScalarKind {.importcpp: "#.scalar_type()".}

# Constructors
# -----------------------------------------------------------------------

# DeviceType and ScalarType are auto-convertible to TensorOptions

func init*(T: type Tensor): Tensor {.constructor, importcpp: "torch::Tensor".}

func from_blob*(data: pointer, sizes: IntArrayRef, options: TensorOptions): Tensor {.importcpp: "torch::from_blob(@)".}
func from_blob*(data: pointer, sizes: IntArrayRef, scalarKind: ScalarKind): Tensor {.importcpp: "torch::from_blob(@)".}
func from_blob*(data: pointer, sizes: IntArrayRef, device: DeviceKind): Tensor {.importcpp: "torch::from_blob(@)".}

func from_blob*(data: pointer, sizes: int64, options: TensorOptions): Tensor {.importcpp: "torch::from_blob(@)".}
func from_blob*(data: pointer, sizes: int64, scalarKind: ScalarKind): Tensor {.importcpp: "torch::from_blob(@)".}
func from_blob*(data: pointer, sizes: int64, device: DeviceKind): Tensor {.importcpp: "torch::from_blob(@)".}

func from_blob*(data: pointer, sizes, strides: IntArrayRef, options: TensorOptions): Tensor {.
    importcpp: "torch::from_blob(@)".}
func from_blob*(data: pointer, sizes, strides: IntArrayRef, scalarKind: ScalarKind): Tensor {.
    importcpp: "torch::from_blob(@)".}
func from_blob*(data: pointer, sizes, strides: IntArrayRef, device: DeviceKind): Tensor {.
    importcpp: "torch::from_blob(@)".}

func empty*(size: IntArrayRef, options: TensorOptions): Tensor {.importcpp: "torch::empty(@)".}
  ## Create an uninitialized tensor of shape `size`
  ## The tensor data must be filled manually
  ##
  ## The output tensor will be row major (C contiguous)
func empty*(size: IntArrayRef, scalarKind: ScalarKind): Tensor {.importcpp: "torch::empty(@)".}
func empty*(size: IntArrayRef, device: DeviceKind): Tensor {.importcpp: "torch::empty(@)".}
  ## Create an uninitialized tensor of shape `size`
  ## The tensor data must be filled manually.
  ##
  ## If device is NOT on CPU make sure to use specialized
  ## copy operations. For example to update on Cuda devices
  ## use cudaMemcpy not a.data[i] = 123
  ##
  ## The output tensor will be row major (C contiguous)

func clone*(self: Tensor): Tensor {.importcpp: "#.clone()".}

# Random sampling
# -----------------------------------------------------------------------

func random_mut*(self: var Tensor, start, stopEx: int64) {.importcpp: "#.random_(@)".}
func randint*(start, stopEx: int64): Tensor {.varargs, importcpp: "torch::randint(#, #, {@})".}
func randint*(start, stopEx: int64, size: IntArrayRef): Tensor {.importcpp: "torch::randint(@)".}

func rand_like*(self: Tensor, options: TensorOptions): Tensor {.importcpp: "torch::rand_like(@)".}
func rand_like*(self: Tensor, options: ScalarKind): Tensor {.importcpp: "torch::rand_like(@)".}
func rand_like*(self: Tensor, options: DeviceKind): Tensor {.importcpp: "torch::rand_like(@)".}
func rand_like*(self: Tensor, options: Device): Tensor {.importcpp: "torch::rand_like(@)".}
func rand_like*(self: Tensor): Tensor {.importcpp: "torch::rand_like(@)".}


# func rand*(size: IntArrayRef, options: TensorOptions): Tensor {.importcpp: "torch::rand(@)"}
func rand*(size: IntArrayRef, options: ScalarKind): Tensor {.importcpp: "torch::rand(@)".}
# func rand*(size: IntArrayRef, options: DeviceKind): Tensor {.importcpp: "torch::rand(@)"}
# func rand*(size: IntArrayRef, options: Device): Tensor {.importcpp: "torch::rand(@)"}
func rand*(size: IntArrayRef): Tensor {.importcpp: "torch::rand(@)".}

# Indexing
# -----------------------------------------------------------------------
# libtorch/include/ATen/TensorIndexing.h
# and https://pytorch.org/cppdocs/notes/tensor_indexing.html

func item*(self: Tensor, T: typedesc): T {.importcpp: "#.item<'0>()".}
  ## Extract the scalar from a 0-dimensional tensor

# Unsure what those corresponds to in Python
# func `[]`*(self: Tensor, index: Scalar): Tensor {.importcpp: "#[#]".}
# func `[]`*(self: Tensor, index: Tensor): Tensor {.importcpp: "#[#]".}
# func `[]`*(self: Tensor, index: int64): Tensor {.importcpp: "#[#]".}

func index*(self: Tensor): Tensor {.varargs, importcpp: "#.index({@})".}
  ## Tensor indexing. It is recommended
  ## to Nimify this in a high-level wrapper.
  ## `tensor.index(indexers)`

# We can't use the construct `#.index_put_({@}, #)`
# so hardcode sizes,
# 6d seems reasonable, that would be a batch of 3D videos (videoID/batchID, Time, Color Channel, Height, Width, Depth)
# If you need more you likely aren't indexing individual values.

func index_put*(self: var Tensor, i0: auto, val: Scalar or Tensor) {.importcpp: "#.index_put_({#}, #)".}
  ## Tensor mutation at index. It is recommended
  ## to Nimify this in a high-level wrapper.
func index_put*(self: var Tensor, i0, i1: auto, val: Scalar or Tensor) {.importcpp: "#.index_put_({#, #}, #)".}
  ## Tensor mutation at index. It is recommended
  ## to Nimify this in a high-level wrapper.
func index_put*(self: var Tensor, i0, i1, i2: auto, val: Scalar or Tensor) {.importcpp: "#.index_put_({#, #, #}, #)".}
  ## Tensor mutation at index. It is recommended
  ## to Nimify this in a high-level wrapper.
func index_put*(self: var Tensor, i0, i1, i2, i3: auto, val: Scalar or Tensor) {.importcpp: "#.index_put_({#, #, #, #}, #)".}
  ## Tensor mutation at index. It is recommended
  ## to Nimify this in a high-level wrapper.
func index_put*(self: var Tensor, i0, i1, i2, i3, i4: auto, val: Scalar or Tensor) {.importcpp: "#.index_put_({#, #, #, #, #}, #)".}
  ## Tensor mutation at index. It is recommended
  ## to Nimify this in a high-level wrapper.
func index_put*(self: var Tensor, i0, i1, i2, i3, i4, i5: auto, val: Scalar or Tensor) {.importcpp: "#.index_put_({#, #, #, #, #, #}, #)".}
  ## Tensor mutation at index. It is recommended
  ## to Nimify this in a high-level wrapper.

# Fancy Indexing
# -----------------------------------------------------------------------

func index_select*(self: Tensor, axis: int64, indices: Tensor): Tensor {.importcpp: "#.index_select(@)".}
func masked_select*(self: Tensor, mask: Tensor): Tensor {.importcpp: "#.masked_select(@)".}

# PyTorch exposes in-place `index_fill_` and `masked_fill_`
# and out-of-place `index_fill` and `masked_fill`
# that does in-place + clone
# we only exposes the in-place version.

func index_fill_mut*(self: var Tensor, mask: Tensor, value: Scalar or Tensor) {.importcpp: "#.index_fill_(@)".}
func masked_fill_mut*(self: var Tensor, mask: Tensor, value: Scalar or Tensor) {.importcpp: "#.masked_fill_(@)".}

# Shapeshifting
# -----------------------------------------------------------------------

func reshape*(self: Tensor): Tensor {.varargs, importcpp: "#.reshape({@})".}
func view*(self: Tensor): Tensor {.varargs, importcpp: "#.reshape({@})".}

# Automatic Differentiation
# -----------------------------------------------------------------------

func backward*(self: var Tensor){.importcpp: "#.backward()".}

# Low-level slicing API
# -----------------------------------------------------------------------

type
  TorchSlice* {.importcpp: "torch::indexing::Slice", bycopy.} = object
  # libtorch/include/ATen/TensorIndexing.h

  TensorIndexType*{.size: sizeof(cint), bycopy, importcpp: "torch::indexing::TensorIndexType".} = enum
    ## This is passed to torchSlice functions
    IndexNone = 0
    IndexEllipsis = 1
    IndexInteger = 2
    IndexBoolean = 3
    IndexSlice = 4
    IndexTensor = 5

  SomeSlicer* = TensorIndexType or SomeSignedInt

proc SliceSpan*(): TorchSlice {.importcpp: "at::indexing::Slice()".}
    ## This is passed to the "index" function
    ## This is Python ":", span / whole dimension

func torchSlice*(){.importcpp: "torch::indexing::Slice(@)", constructor.}
func torchSlice*(start: SomeSlicer): TorchSlice {.importcpp: "torch::indexing::Slice(@)", constructor.}
func torchSlice*(start: SomeSlicer, stop: SomeSlicer): TorchSlice {.importcpp: "torch::indexing::Slice(@)", constructor.}
func torchSlice*(start: SomeSlicer, stop: SomeSlicer, step: SomeSlicer): TorchSlice {.importcpp: "torch::indexing::Slice(@)", constructor.}
func start*(s: TorchSlice): int64 {.importcpp: "#.start()".}
func stop*(s: TorchSlice): int64 {.importcpp: "#.stop()".}
func step*(s: TorchSlice): int64 {.importcpp: "#.step()".}

# Operators
# -----------------------------------------------------------------------

func `not`*(self: Tensor): Tensor {.importcpp: "(~#)".}
func `-`*(self: Tensor): Tensor {.importcpp: "(-#)".}

func `+`*(self: Tensor, b: Tensor): Tensor {.importcpp: "(# + #)".}
func `-`*(self: Tensor, b: Tensor): Tensor {.importcpp: "(# - #)".}
func `*`*(self: Tensor, b: Tensor): Tensor {.importcpp: "(# * #)".}

func `*`*(a: cfloat or cdouble, b: Tensor): Tensor {.importcpp: "(# * #)".}
func `*`*(self: Tensor, b: cfloat or cdouble): Tensor {.importcpp: "(# * #)".}

func `+=`*(self: var Tensor, b: Tensor) {.importcpp: "(# += #)".}
func `+=`*(self: var Tensor, s: Scalar) {.importcpp: "(# += #)".}
func `-=`*(self: var Tensor, b: Tensor) {.importcpp: "(# -= #)".}
func `-=`*(self: var Tensor, s: Scalar) {.importcpp: "(# -= #)".}
func `*=`*(self: var Tensor, b: Tensor) {.importcpp: "(# *= #)".}
func `*=`*(self: var Tensor, s: Scalar) {.importcpp: "(# *= #)".}
func `/=`*(self: var Tensor, b: Tensor) {.importcpp: "(# /= #)".}
func `/=`*(self: var Tensor, s: Scalar) {.importcpp: "(# /= #)".}

func `and`*(self: Tensor, b: Tensor): Tensor {.importcpp: "#.bitwise_and(#)".}
  ## bitwise `and`.
func `or`*(self: Tensor, b: Tensor): Tensor {.importcpp: "#.bitwise_or(#)".}
  ## bitwise `or`.
func `xor`*(self: Tensor, b: Tensor): Tensor {.importcpp: "#.bitwise_xor(#)".}
  ## bitwise `xor`.

func bitand_mut*(self: var Tensor, s: Tensor) {.importcpp: "#.bitwise_and_(#)".}
  ## In-place bitwise `and`.
func bitor_mut*(self: var Tensor, s: Tensor) {.importcpp: "#.bitwise_or_(#)".}
  ## In-place bitwise `or`.
func bitxor_mut*(self: var Tensor, s: Tensor) {.importcpp: "#.bitwise_xor_(#)".}
  ## In-place bitwise `xor`.

func eq*(a, b: Tensor): Tensor {.importcpp: "#.eq(#)".}
  ## Equality of each tensor values
func equal*(a, b: Tensor): bool {.importcpp: "#.equal(#)".}
template `==`*(a, b: Tensor): bool =
  a.equal(b)

# Functions.h
# -----------------------------------------------------------------------

func toType*(self: Tensor, dtype: ScalarKind): Tensor {.importcpp: "#.toType(@)".}
func toSparse*(self: Tensor): Tensor {.importcpp: "#.to_sparse()".}
func toSparse*(self: Tensor, sparseDim: int64): Tensor {.importcpp: "#.to_sparse(@)".}

func eye*(n: int64): Tensor {.importcpp: "torch::eye(@)".}
func eye*(n: int64, options: TensorOptions): Tensor {.importcpp: "torch::eye(@)".}
func eye*(n: int64, scalarKind: ScalarKind): Tensor {.importcpp: "torch::eye(@)".}
func eye*(n: int64, device: DeviceKind): Tensor {.importcpp: "torch::eye(@)".}

func zeros*(dim: int64): Tensor {.importcpp: "torch::zeros(@)".}
func zeros*(dim: IntArrayRef): Tensor {.importcpp: "torch::zeros(@)".}
func zeros*(dim: IntArrayRef, options: TensorOptions): Tensor {.importcpp: "torch::zeros(@)".}
func zeros*(dim: IntArrayRef, scalarKind: ScalarKind): Tensor {.importcpp: "torch::zeros(@)".}
func zeros*(dim: IntArrayRef, device: DeviceKind): Tensor {.importcpp: "torch::zeros(@)".}

func linspace*(start, stop: Scalar, steps: int64, options: TensorOptions) : Tensor {.importcpp: "#.linspace(@)".}
func linspace*(start, stop: Scalar, steps: int64, options: ScalarKind) : Tensor {.importcpp: "#.linspace(@)".}
func linspace*(start, stop: Scalar, steps: int64, options: DeviceKind) : Tensor {.importcpp: "#.linspace(@)".}
func linspace*(start, stop: Scalar, steps: int64, options: Device) : Tensor {.importcpp: "#.linspace(@)".}
func linspace*(start, stop: Scalar, steps: int64) : Tensor {.importcpp: "torch::linspace(@)".}
func linspace*(start, stop: Scalar) : Tensor {.importcpp: "torch::linspace(@)".}

func logspace*(start, stop: Scalar, steps, base: int64, options: TensorOptions) : Tensor {.importcpp: "torch::logspace(@)".}
func logspace*(start, stop: Scalar, steps, base: int64, options: ScalarKind) : Tensor {.importcpp: "torch::logspace(@)".}
func logspace*(start, stop: Scalar, steps, base: int64, options: DeviceKind) {.importcpp: "torch::logspace(@)".}
func logspace*(start, stop: Scalar, steps, base: int64, options: Device)  : Tensor {.importcpp: "torch::logspace(@)".}
func logspace*(start, stop: Scalar, steps, base: int64) : Tensor {.importcpp: "torch::logspace(@)".}
func logspace*(start, stop: Scalar, steps: int64)  : Tensor {.importcpp: "torch::logspace(@)".}
func logspace*(start, stop: Scalar)  : Tensor {.importcpp: "torch::logspace(@)".}

func arange*(stop: Scalar, options: TensorOptions) : Tensor  {.importcpp: "torch::arange(@)".}
func arange*(stop: Scalar, options: ScalarKind) : Tensor  {.importcpp: "torch::arange(@)".}
func arange*(stop: Scalar, options: DeviceKind) : Tensor  {.importcpp: "torch::arange(@)".}
func arange*(stop: Scalar, options: Device) : Tensor  {.importcpp: "torch::arange(@)".}
func arange*(stop: Scalar) : Tensor  {.importcpp: "torch::arange(@)".}
func arange*(start, stop: Scalar, options: TensorOptions) : Tensor  {.importcpp: "torch::arange(@)".}
func arange*(start, stop: Scalar, options: ScalarKind) : Tensor  {.importcpp: "torch::arange(@)".}
func arange*(start, stop: Scalar, options: DeviceKind) : Tensor  {.importcpp: "torch::arange(@)".}
func arange*(start, stop: Scalar, options: Device) : Tensor  {.importcpp: "torch::arange(@)".}
func arange*(start, stop: Scalar) : Tensor  {.importcpp: "torch::arange(@)".}
func arange*(start, stop, step: Scalar, options: TensorOptions) : Tensor  {.importcpp: "torch::arange(@)".}
func arange*(start, stop, step: Scalar, options: ScalarKind) : Tensor  {.importcpp: "torch::arange(@)".}
func arange*(start, stop, step: Scalar, options: DeviceKind) : Tensor  {.importcpp: "torch::arange(@)".}
func arange*(start, stop, step: Scalar, options: Device) : Tensor  {.importcpp: "torch::arange(@)".}
func arange*(start, stop, step: Scalar) : Tensor  {.importcpp: "torch::arange(@)".}

# Operations
# -----------------------------------------------------------------------
func add*(self: Tensor, other: Tensor, alpha: Scalar = 1): Tensor {.importcpp: "#.add(@)".}
func add*(self: Tensor, other: Scalar, alpha: Scalar = 1): Tensor {.importcpp: "#.add(@)".}
func addmv*(self: Tensor, mat: Tensor, vec: Tensor, beta: Scalar = 1, alpha: Scalar = 1): Tensor {.importcpp: "#.addmv(@)".}
func addmm*(t, mat1, mat2: Tensor, beta: Scalar = 1, alpha: Scalar = 1): Tensor {.importcpp: "#.addmm(@)".}
func mm*(t, other: Tensor): Tensor {.importcpp: "#.mm(@)".}
func matmul*(t, other: Tensor): Tensor {.importcpp: "#.matmul(@)".}
func bmm*(t, other: Tensor): Tensor {.importcpp: "#.bmm(@)".}

func luSolve*(t, data, pivots: Tensor): Tensor {.importcpp: "#.lu_solve(@)".}

func qr*(self: Tensor, some: bool = true): CppTuple2[Tensor, Tensor] {.importcpp: "#.qr(@)".}
  ## Returns a tuple:
  ## - Q of shape (∗,m,k)
  ## - R of shape (∗,k,n)
  ## with k=min(m,n) if some is true otherwise k=m
  ##
  ## The QR decomposition is batched over dimension(s) *
  ## t = QR

# addr?
func all*(self: Tensor, axis: int64): Tensor {.importcpp: "#.all(@)".}
func all*(self: Tensor, axis: int64, keepdim: bool): Tensor {.importcpp: "#.all(@)".}
func allClose*(t, other: Tensor, rtol: float64 = 1e-5, abstol: float64 = 1e-8, equalNan: bool = false): bool {.importcpp: "#.allclose(@)".}
func any*(self: Tensor, axis: int64): Tensor {.importcpp: "#.any(@)".}
func any*(self: Tensor, axis: int64, keepdim: bool): Tensor {.importcpp: "#.any(@)".}
func argmax*(self: Tensor): Tensor {.importcpp: "#.argmax()".}
func argmax*(self: Tensor, axis: int64, keepdim: bool = false): Tensor {.importcpp: "#.argmax(@)".}
func argmin*(self: Tensor): Tensor {.importcpp: "#.argmin()".}
func argmin*(self: Tensor, axis: int64, keepdim: bool = false): Tensor {.importcpp: "#.argmin(@)".}

# aggregate
# -----------------------------------------------------------------------

# sum needs wrapper procs/templates to allow for using nim arrays and single axis.
func sum*(self: Tensor): Tensor {.importcpp: "#.sum()".}
func sum*(self: Tensor, dtype: ScalarKind): Tensor {.importcpp: "#.sum(@)".}
func sum*(self: Tensor, axis: int64, keepdim: bool = false): Tensor {.importcpp: "#.sum(@)".}
func sum*(self: Tensor, axis: int64, keepdim: bool = false, dtype: ScalarKind): Tensor {.importcpp: "#.sum(@)".}
func sum*(self: Tensor, axis: IntArrayRef, keepdim: bool = false): Tensor {.importcpp: "#.sum(@)".}
func sum*(self: Tensor, axis: IntArrayRef, keepdim: bool = false, dtype: ScalarKind): Tensor {.importcpp: "#.sum(@)".}

# mean as well
func mean*(self: Tensor): Tensor {.importcpp: "#.mean()".}
func mean*(self: Tensor, dtype: ScalarKind): Tensor {.importcpp: "#.mean(@)".}
func mean*(self: Tensor, axis: int64, keepdim: bool = false): Tensor {.importcpp: "#.mean(@)".}
func mean*(self: Tensor, axis: int64, keepdim: bool = false, dtype: ScalarKind): Tensor {.importcpp: "#.mean(@)".}
func mean*(self: Tensor, axis: IntArrayRef, keepdim: bool = false): Tensor {.importcpp: "#.mean(@)".}
func mean*(self: Tensor, axis: IntArrayRef, keepdim: bool = false, dtype: ScalarKind): Tensor {.importcpp: "#.mean(@)".}

# median requires std::tuple

func prod*(self: Tensor): Tensor {.importcpp: "#.prod()".}
func prod*(self: Tensor, dtype: ScalarKind): Tensor {.importcpp: "#.prod(@)".}
func prod*(self: Tensor, axis: int64, keepdim: bool = false): Tensor {.importcpp: "#.prod(@)".}
func prod*(self: Tensor, axis: int64, keepdim: bool = false, dtype: ScalarKind): Tensor {.importcpp: "#.prod(@)".}

func min*(self: Tensor): Tensor {.importcpp: "#.min()".}
func min*(self: Tensor, axis: int64, keepdim: bool = false): CppTuple2[Tensor, Tensor] {.importcpp: "torch::min(@)".}
  ## Returns a tuple (values, indices) of type (TensorT, TensorInt64)
  ## of the minimum values and their index in the specified axis

func max*(self: Tensor): Tensor {.importcpp: "#.max()".}
func max*(self: Tensor, axis: int64, keepdim: bool = false): CppTuple2[Tensor, Tensor] {.importcpp: "torch::max(@)".}
  ## Returns a tuple (values, indices) of type (TensorT, TensorInt64)
  ## of the maximum values and their index in the specified axis

func variance*(self: Tensor, unbiased: bool = true): Tensor {.importcpp: "#.var(@)".} # can't use `var` because of keyword.
func variance*(self: Tensor, axis: int64, unbiased: bool = true, keepdim: bool = false): Tensor {.importcpp: "#.var(@)".}
func variance*(self: Tensor, axis: IntArrayRef, unbiased: bool = true, keepdim: bool = false): Tensor {.importcpp: "#.var(@)".}

func stddev*(self: Tensor, unbiased: bool = true): Tensor {.importcpp: "#.std(@)".}
func stddev*(self: Tensor, axis: int64, unbiased: bool = true, keepdim: bool = false): Tensor {.importcpp: "#.std(@)".}
func stddev*(self: Tensor, axis: IntArrayRef, unbiased: bool = true, keepdim: bool = false): Tensor {.importcpp: "#.std(@)".}

# algorithms:
# -----------------------------------------------------------------------

func sort*(self: Tensor, axis: int64 = -1, descending: bool = false): CppTuple2[Tensor, Tensor] {.importcpp: "#.sort(@)".}
  ## Sorts the elements of the input tensor along a given dimension in ascending order by value.
  ## If dim is not given, the last dimension of the input is chosen (dim=-1).
  ## Returns (values, originalIndices) or type (TensorT, TensorInt64)
  ## where originalIndices is the original index of each values (before sorting)
func argsort*(self: Tensor, axis: int64 = -1, descending: bool = false): Tensor {.importcpp: "#.argsort(@)".}

# math
# -----------------------------------------------------------------------
func abs*(self: Tensor): Tensor {.importcpp: "#.abs()".}
func absolute*(self: Tensor): Tensor {.importcpp: "#.absolute()".}
func angle*(self: Tensor): Tensor {.importcpp: "#.angle()".}
func sgn*(self: Tensor): Tensor {.importcpp: "#.sgn()".}
func conj*(self: Tensor): Tensor {.importcpp: "#.conj()".}
func acos*(self: Tensor): Tensor {.importcpp: "#.acos()".}
func arccos*(self: Tensor): Tensor {.importcpp: "#.arccos()".}
func acosh*(self: Tensor): Tensor {.importcpp: "#.acosh()".}
func arccosh*(self: Tensor): Tensor {.importcpp: "#.arccosh()".}
func asinh*(self: Tensor): Tensor {.importcpp: "#.asinh()".}
func arcsinh*(self: Tensor): Tensor {.importcpp: "#.arcsinh()".}
func atanh*(self: Tensor): Tensor {.importcpp: "#.atanh()".}
func arctanh*(self: Tensor): Tensor {.importcpp: "#.arctanh()".}
func asin*(self: Tensor): Tensor {.importcpp: "#.asin()".}
func arcsin*(self: Tensor): Tensor {.importcpp: "#.arcsin()".}
func atan*(self: Tensor): Tensor {.importcpp: "#.atan()".}
func arctan*(self: Tensor): Tensor {.importcpp: "#.arctan()".}
func cos*(self: Tensor): Tensor {.importcpp: "#.cos()".}
func sin*(self: Tensor): Tensor {.importcpp: "#.sin()".}
func tan*(self: Tensor): Tensor {.importcpp: "#.tan()".}
func exp*(self: Tensor): Tensor {.importcpp: "#.exp()".}
func exp2*(self: Tensor): Tensor {.importcpp: "#.exp2()".}
func erf*(self: Tensor): Tensor {.importcpp: "#.erf()".}
func erfc*(self: Tensor): Tensor {.importcpp: "#.erfc()".}
func reciprocal*(self: Tensor): Tensor {.importcpp: "#.reciprocal()".}
func neg*(self: Tensor): Tensor {.importcpp: "#.neg()".}
func clamp*(self: Tensor, min, max: Scalar): Tensor {.importcpp: "#.clamp(@)".}
func clampMin*(self: Tensor, min: Scalar): Tensor {.importcpp: "#.clamp_min(@)".}
func clampMax*(self: Tensor, max: Scalar): Tensor {.importcpp: "#.clamp_max(@)".}

func dot*(self: Tensor, other: Tensor): Tensor {.importcpp: "#.dot(@)".}

func squeeze*(self: Tensor): Tensor {.importcpp: "#.squeeze()".}
func squeeze*(self: Tensor, axis: int64): Tensor {.importcpp: "#.squeeze(@)".}
func unsqueeze*(self: Tensor, axis: int64): Tensor {.importcpp: "#.unsqueeze(@)".}

# FFT
# -----------------------------------------------------------------------
func fftshift*(self: Tensor): Tensor {.importcpp: "torch::fft_fftshift(@)".}
func fftshift*(self: Tensor, dim: IntArrayRef): Tensor {.importcpp: "torch::fft_ifftshift(@)".}
func ifftshift*(self: Tensor): Tensor {.importcpp: "torch::fft_fftshift(@)".}
func ifftshift*(self: Tensor, dim: IntArrayRef): Tensor {.importcpp: "torch::fft_ifftshift(@)".}

let defaultNorm : CppString = initCppString("backward")

func fft*(self: Tensor, n: int64, dim: int64 = -1, norm: CppString = defaultNorm): Tensor {.importcpp: "torch::fft_fft(@)".}
## Compute the 1-D Fourier transform
## ``n`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
## ``norm`` can be :
##    * "forward" - normalize by 1/n
##    * "backward" - no normalization
##    * "ortho" - normalize by 1/sqrt(n)
func fft*(self: Tensor, dim: int64 = -1, norm: CppString = defaultNorm): Tensor {.importcpp: "torch::fft_fft(@)".}
## Compute the 1-D Fourier transform

func ifft*(self: Tensor, n: int64, dim: int64 = -1, norm: CppString = defaultNorm): Tensor {.importcpp: "torch::ifft_ifft(@)".}
## Compute the 1-D Fourier transform
## ``norm`` can be :
##   * "forward" - no normalization
##   * "backward" - normalization by 1/n
##   * "ortho" - normalization by 1/sqrt(n)
func ifft*(self: Tensor, dim: int64 = -1, norm: CppString = defaultNorm): Tensor {.importcpp: "torch::ifft_ifft(@)".}
## Compute the 1-D Fourier transform

func fft2*(self: Tensor, s: IntArrayRef, dim: IntArrayRef, norm: CppString = defaultNorm): Tensor {.
    importcpp: "torch::fft_fft2(@)".}
## Compute the 2-D Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
## ``norm`` can be :
##    * "forward" - normalize by 1/n
##    * "backward" - no normalization
##    * "ortho" - normalize by 1/sqrt(n)
## With n the logical FFT size: ``n = prod(s)``.
func fft2*(self: Tensor, s: IntArrayRef): Tensor {.importcpp: "torch::fft_fft2(@)".}
## Compute the 2-D Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
func fft2*(self: Tensor): Tensor {.importcpp: "torch::fft_fft2(@)".}
## Compute the 2-D Fourier transform

func ifft2*(self: Tensor, s: IntArrayRef, dim: IntArrayRef, norm: CppString = defaultNorm): Tensor {.
    importcpp: "torch::fft_ifft2(@)".}
## Compute the 2-D Inverse Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
## ``norm`` can be :
##   * "forward" - no normalization
##   * "backward" - normalization by 1/n
##   * "ortho" - normalization by 1/sqrt(n)
## With n the logical FFT size: ``n = prod(s)``.
func ifft2*(self: Tensor, s: IntArrayRef): Tensor {.importcpp: "torch::fft_ifft2(@)".}
## Compute the 2-D Inverse Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
func ifft2*(self: Tensor): Tensor {.importcpp: "torch::fft_ifft2(@)".}
## Compute the 2-D Inverse Fourier transform

func fftn*(self: Tensor, s: IntArrayRef, dim: IntArrayRef, norm: CppString = defaultNorm): Tensor {.importcpp: "torch::fft_fftn(@)".}
## Compute the N-D Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
## ``norm`` can be :
##    * "forward" normalize by 1/n
##    * "backward" - no normalization
##    * "ortho" normalize by 1/sqrt(n)
## With n the logical FFT size: ``n = prod(s)``.
func fftn*(self: Tensor, s: IntArrayRef): Tensor {.importcpp: "torch::fft_fftn(@)".}
## Compute the N-D Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
func fftn*(self: Tensor): Tensor {.importcpp: "torch::fft_fftn(@)".}
## Compute the N-D Fourier transform

func ifftn*(self: Tensor, s: IntArrayRef, dim: IntArrayRef, norm: CppString = defaultNorm): Tensor {.importcpp: "torch::fft_ifftn(@)".}
## Compute the N-D Inverse Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
## ``norm`` can be :
##   * "forward" - no normalization
##   * "backward" - normalization by 1/n
##   * "ortho" - normalization by 1/sqrt(n)
## With n the logical FFT size: ``n = prod(s)``.
func ifftn*(self: Tensor, s: IntArrayRef): Tensor {.importcpp: "torch::fft_ifftn(@)".}
## Compute the N-D Inverse Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
func ifftn*(self: Tensor): Tensor {.importcpp: "torch::fft_ifftn(@)".}
## Compute the N-D Inverse Fourier transform

func rfft*(self: Tensor, n: int64, dim: int64 = -1, norm: CppString = defaultNorm): Tensor {.importcpp: "torch::fft_rfft".}
## Computes the one dimensional Fourier transform of real-valued input.
func rfft*(self: Tensor, dim: int64 = -1, norm: CppString = defaultNorm): Tensor {.importcpp: "torch::fft_rfft".}
## Computes the one dimensional Fourier transform of real-valued input.

func irfft*(self: Tensor, n: int64, dim: int64 = -1, norm: CppString = defaultNorm): Tensor {.importcpp: "torch::fft_irfft".}
## Computes the one dimensional Fourier transform of real-valued input.
func irfft*(self: Tensor, dim: int64 = -1, norm: CppString = defaultNorm): Tensor {.importcpp: "torch::fft_irfft".}
## Computes the one dimensional Fourier transform of real-valued input.

func rfft2*(self: Tensor, s: IntArrayRef, dim: IntArrayRef, norm: CppString = defaultNorm): Tensor {.importcpp: "torch::fft_rfft2(@)".}
## Compute the N-D Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
## ``norm`` can be :
##    * "forward" - normalize by 1/n
##    * "backward" - no normalization
##    * "ortho" - normalize by 1/sqrt(n)
## With n the logical FFT size: ``n = prod(s)``.
func rfft2*(self: Tensor, s: IntArrayRef): Tensor {.importcpp: "torch::fft_rfft2(@)".}
## Compute the N-D Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
func rfft2*(self: Tensor): Tensor {.importcpp: "torch::fft_rfft2(@)".}
## Compute the N-D Fourier transform

func irfft2*(self: Tensor, s: IntArrayRef, dim: IntArrayRef, norm: CppString = defaultNorm): Tensor {.
    importcpp: "torch::fft_irfft2(@)".}
## Compute the N-D Inverse Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
## ``norm`` can be :
##   * "forward" - no normalization
##   * "backward" - normalization by 1/n
##   * "ortho" - normalization by 1/sqrt(n)
## With n the logical FFT size: ``n = prod(s)``.
func irfft2*(self: Tensor, s: IntArrayRef): Tensor {.importcpp: "torch::fft_irfft2(@)".}
## Compute the N-D Inverse Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
func irfft2*(self: Tensor): Tensor {.importcpp: "torch::fft_irfft2(@)".}
## Compute the N-D Inverse Fourier transform


func rfftn*(self: Tensor, s: IntArrayRef, dim: IntArrayRef, norm: CppString = defaultNorm): Tensor {.importcpp: "torch::fft_rfftn(@)".}
## Compute the N-D Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
## ``norm`` can be :
##    * "forward" - normalize by 1/n
##    * "backward" - no normalization
##    * "ortho" - normalize by 1/sqrt(n)
## With n the logical FFT size: ``n = prod(s)``.
func rfftn*(self: Tensor, s: IntArrayRef): Tensor {.importcpp: "torch::fft_rfftn(@)".}
## Compute the N-D Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
func rfftn*(self: Tensor): Tensor {.importcpp: "torch::fft_rfftn(@)".}
## Compute the N-D Fourier transform

func irfftn*(self: Tensor, s: IntArrayRef, dim: IntArrayRef, norm: CppString = defaultNorm): Tensor {.
    importcpp: "torch::fft_irfftn(@)".}
## Compute the N-D Inverse Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
## ``norm`` can be :
##   * "forward" - no normalization
##   * "backward" - normalization by 1/n
##   * "ortho" - normalization by 1/sqrt(n)
## With n the logical FFT size: ``n = prod(s)``.
func irfftn*(self: Tensor, s: IntArrayRef): Tensor {.importcpp: "torch::fft_irfftn(@)".}
## Compute the N-D Inverse Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
func irfftn*(self: Tensor): Tensor {.importcpp: "torch::fft_irfftn(@)".}
## Compute the N-D Inverse Fourier transform

func hfft*(self: Tensor, n: int64, dim: int64 = -1, norm: CppString = defaultNorm) : Tensor {.importcpp: "torch::hfft".}
## Computes the 1 dimensional FFT of a onesided Hermitian signal.
func hfft*(self: Tensor, dim: int64 = -1, norm: CppString = defaultNorm) : Tensor {.importcpp: "torch::hfft".}
## Computes the 1 dimensional FFT of a onesided Hermitian signal.
func ihfft*(self: Tensor, n: int64, dim: int64 = -1, norm: CppString = defaultNorm) : Tensor {.importcpp: "torch::ihfft".}
## Computes the inverse FFT of a real-valued Fourier domain signal.
func ihfft*(self: Tensor, dim: int64 = -1, norm: CppString = defaultNorm) : Tensor {.importcpp: "torch::ihfft".}
## Computes the inverse FFT of a real-valued Fourier domain signal.

#func convolution*(self: Tensor, weight: Tensor, bias: Tensor, stride, padding, dilation: int64, transposed: bool, outputPadding: int64, groups: int64): Tensor {.importcpp: "torch::convolution(@)".}
