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
  ../../libtorch,
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
  ATTensor* {.importcpp: "torch::Tensor", bycopy.} = object

# Strings & Debugging
# -----------------------------------------------------------------------

proc print*(self: ATTensor) {.sideeffect, importcpp: "torch::print(@)".}

# Metadata
# -----------------------------------------------------------------------

func dim*(self: ATTensor): int64 {.importcpp: "#.dim()".}
  ## Number of dimensions
func reset*(self: var ATTensor) {.importcpp: "#.reset()".}
func is_same*(self, other: ATTensor): bool {.importcpp: "#.is_same(#)".}
  ## Reference equality
  ## Do the tensors use the same memory.

func sizes*(self: ATTensor): IntArrayRef {.importcpp: "#.sizes()".}
  ## This is Arraymancer and Numpy "shape"

func strides*(self: ATTensor): IntArrayRef {.importcpp: "#.strides()".}

func ndimension*(self: ATTensor): int64 {.importcpp: "#.ndimension()".}
  ## This is Arraymancer rank
func nbytes*(self: ATTensor): uint {.importcpp: "#.nbytes()".}
  ## Bytes-size of the Tensor
func numel*(self: ATTensor): int64 {.importcpp: "#.numel()".}
  ## This is Arraymancer and Numpy "size"

func size*(self: ATTensor, axis: int64): int64 {.importcpp: "#.size(#)".}
func itemsize*(self: ATTensor): uint {.importcpp: "#.itemsize()".}
func element_size*(self: ATTensor): int64 {.importcpp: "#.element_size()".}

# Accessors
# -----------------------------------------------------------------------

func data_ptr*(self: ATTensor, T: typedesc): ptr UncheckedArray[T] {.importcpp: "#.data_ptr<'2>(#)".}
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

func has_storage*(self: ATTensor): bool {.importcpp: "#.has_storage()".}
func get_device*(self: ATTensor): int64 {.importcpp: "#.get_device()".}
func is_cuda*(self: ATTensor): bool {.importcpp: "#.is_cuda()".}
func is_hip*(self: ATTensor): bool {.importcpp: "#.is_hip()".}
func is_sparse*(self: ATTensor): bool {.importcpp: "#.is_sparse()".}
func is_mkldnn*(self: ATTensor): bool {.importcpp: "#.is_mkldnn()".}
func is_vulkan*(self: ATTensor): bool {.importcpp: "#.is_vulkan()".}
func is_quantized*(self: ATTensor): bool {.importcpp: "#.is_quantized()".}
func is_meta*(self: ATTensor): bool {.importcpp: "#.is_meta()".}

func cpu*(self: ATTensor): ATTensor {.importcpp: "#.cpu()".}
func cuda*(self: ATTensor): ATTensor {.importcpp: "#.cuda()".}
func hip*(self: ATTensor): ATTensor {.importcpp: "#.hip()".}
func vulkan*(self: ATTensor): ATTensor {.importcpp: "#.vulkan()".}
func to*(self: ATTensor, device: DeviceKind): ATTensor {.importcpp: "#.to(#)".}
func to*(self: ATTensor, device: Device): ATTensor {.importcpp: "#.to(#)".}

# dtype
# -----------------------------------------------------------------------

func to*(self: ATTensor, dtype: ScalarKind): ATTensor {.importcpp: "#.to(#)".}
func scalarType*(self: ATTensor): ScalarKind {.importcpp: "#.scalar_type()".}

# Constructors
# -----------------------------------------------------------------------

# DeviceType and ScalarType are auto-convertible to TensorOptions

func init*(T: type ATTensor): ATTensor {.constructor, importcpp: "torch::Tensor".}

func from_blob*(data: pointer, sizes: IntArrayRef, options: TensorOptions): ATTensor {.importcpp: "torch::from_blob(@)".}
func from_blob*(data: pointer, sizes: IntArrayRef, scalarKind: ScalarKind): ATTensor {.importcpp: "torch::from_blob(@)".}
func from_blob*(data: pointer, sizes: IntArrayRef, device: DeviceKind): ATTensor {.importcpp: "torch::from_blob(@)".}

func from_blob*(data: pointer, sizes: int64, options: TensorOptions): ATTensor {.importcpp: "torch::from_blob(@)".}
func from_blob*(data: pointer, sizes: int64, scalarKind: ScalarKind): ATTensor {.importcpp: "torch::from_blob(@)".}
func from_blob*(data: pointer, sizes: int64, device: DeviceKind): ATTensor {.importcpp: "torch::from_blob(@)".}

func from_blob*(data: pointer, sizes, strides: IntArrayRef, options: TensorOptions): ATTensor {.
    importcpp: "torch::from_blob(@)".}
func from_blob*(data: pointer, sizes, strides: IntArrayRef, scalarKind: ScalarKind): ATTensor {.
    importcpp: "torch::from_blob(@)".}
func from_blob*(data: pointer, sizes, strides: IntArrayRef, device: DeviceKind): ATTensor {.
    importcpp: "torch::from_blob(@)".}

func empty*(size: IntArrayRef, options: TensorOptions): ATTensor {.importcpp: "torch::empty(@)".}
  ## Create an uninitialized tensor of shape `size`
  ## The tensor data must be filled manually
  ##
  ## The output tensor will be row major (C contiguous)
func empty*(size: IntArrayRef, scalarKind: ScalarKind): ATTensor {.importcpp: "torch::empty(@)".}
func empty*(size: IntArrayRef, device: DeviceKind): ATTensor {.importcpp: "torch::empty(@)".}
  ## Create an uninitialized tensor of shape `size`
  ## The tensor data must be filled manually.
  ##
  ## If device is NOT on CPU make sure to use specialized
  ## copy operations. For example to update on Cuda devices
  ## use cudaMemcpy not a.data[i] = 123
  ##
  ## The output tensor will be row major (C contiguous)

func clone*(self: ATTensor): ATTensor {.importcpp: "#.clone()".}

# Random sampling
# -----------------------------------------------------------------------

func random_mut*(self: var ATTensor, start, stopEx: int64) {.importcpp: "#.random_(@)".}
func randint*(start, stopEx: int64): ATTensor {.varargs, importcpp: "torch::randint(#, #, {@})".}
func randint*(start, stopEx: int64, size: IntArrayRef): ATTensor {.importcpp: "torch::randint(@)".}

func rand_like*(self: ATTensor, options: TensorOptions): ATTensor {.importcpp: "torch::rand_like(@)".}
func rand_like*(self: ATTensor, options: ScalarKind): ATTensor {.importcpp: "torch::rand_like(@)".}
func rand_like*(self: ATTensor, options: DeviceKind): ATTensor {.importcpp: "torch::rand_like(@)".}
func rand_like*(self: ATTensor, options: Device): ATTensor {.importcpp: "torch::rand_like(@)".}
func rand_like*(self: ATTensor): ATTensor {.importcpp: "torch::rand_like(@)".}

func rand*(size: IntArrayRef, options: TensorOptions): ATTensor {.importcpp: "torch::rand(@)"}
func rand*(size: IntArrayRef, options: DeviceKind): ATTensor {.importcpp: "torch::rand(@)"}
func rand*(size: IntArrayRef, options: Device): ATTensor {.importcpp: "torch::rand(@)"}
func rand*(size: IntArrayRef, options: ScalarKind): ATTensor {.importcpp: "torch::rand(@)".}
func rand*(size: IntArrayRef): ATTensor {.importcpp: "torch::rand(@)".}

# Indexing
# -----------------------------------------------------------------------
# TODO throw IndexDefect when bounds checking is active
# libtorch/include/ATen/TensorIndexing.h
# and https://pytorch.org/cppdocs/notes/tensor_indexing.html

func item*(self: ATTensor, T: typedesc): T {.importcpp: "#.item<'0>()".}
  ## Extract the scalar from a 0-dimensional tensor
func item*(self: ATTensor, T: typedesc[Complex32]): C10_Complex[float32] {.importcpp: "#.item<c10::complex<float>>()".}
func item*(self: ATTensor, T: typedesc[Complex64]): C10_Complex[float64] {.importcpp: "#.item<c10::complex<double>>()".}


# Unsure what those corresponds to in Python
# func `[]`*(self: Tensor, index: Scalar): Tensor {.importcpp: "#[#]".}
# func `[]`*(self: Tensor, index: Tensor): Tensor {.importcpp: "#[#]".}
# func `[]`*(self: Tensor, index: int64): Tensor {.importcpp: "#[#]".}

func index*(self: ATTensor): ATTensor {.varargs, importcpp: "#.index({@})".}
  ## Tensor indexing. It is recommended
  ## to Nimify this in a high-level wrapper.
  ## `tensor.index(indexers)`

# We can't use the construct `#.index_put_({@}, #)`
# so hardcode sizes,
# 6d seems reasonable, that would be a batch of 3D videos (videoID/batchID, Time, Color Channel, Height, Width, Depth)
# If you need more you likely aren't indexing individual values.

func index_put*(self: var ATTensor, i0: auto, val: Scalar or ATTensor) {.importcpp: "#.index_put_({#}, #)".}
  ## Tensor mutation at index. It is recommended
  ## to Nimify this in a high-level wrapper.
func index_put*(self: var ATTensor, i0, i1: auto, val: Scalar or ATTensor) {.importcpp: "#.index_put_({#, #}, #)".}
  ## Tensor mutation at index. It is recommended
  ## to Nimify this in a high-level wrapper.
func index_put*(self: var ATTensor, i0, i1, i2: auto, val: Scalar or ATTensor) {.importcpp: "#.index_put_({#, #, #}, #)".}
  ## Tensor mutation at index. It is recommended
  ## to Nimify this in a high-level wrapper.
func index_put*(self: var ATTensor, i0, i1, i2, i3: auto, val: Scalar or ATTensor) {.importcpp: "#.index_put_({#, #, #, #}, #)".}
  ## Tensor mutation at index. It is recommended
  ## to Nimify this in a high-level wrapper.
func index_put*(self: var ATTensor, i0, i1, i2, i3, i4: auto, val: Scalar or ATTensor) {.importcpp: "#.index_put_({#, #, #, #, #}, #)".}
  ## Tensor mutation at index. It is recommended
  ## to Nimify this in a high-level wrapper.
func index_put*(self: var ATTensor, i0, i1, i2, i3, i4, i5: auto, val: Scalar or ATTensor) {.importcpp: "#.index_put_({#, #, #, #, #, #}, #)".}
  ## Tensor mutation at index. It is recommended
  ## to Nimify this in a high-level wrapper.

# Fancy Indexing
# -----------------------------------------------------------------------

func index_select*(self: ATTensor, axis: int64, indices: ATTensor): ATTensor {.importcpp: "#.index_select(@)".}
func masked_select*(self: ATTensor, mask: ATTensor): ATTensor {.importcpp: "#.masked_select(@)".}

# PyTorch exposes in-place `index_fill_` and `masked_fill_`
# and out-of-place `index_fill` and `masked_fill`
# that does in-place + clone
# we only exposes the in-place version.

func index_fill_mut*(self: var ATTensor, mask: ATTensor, value: Scalar or ATTensor) {.importcpp: "#.index_fill_(@)".}
func masked_fill_mut*(self: var ATTensor, mask: ATTensor, value: Scalar or ATTensor) {.importcpp: "#.masked_fill_(@)".}

# Shapeshifting
# -----------------------------------------------------------------------

func reshape*(self: ATTensor, sizes: IntArrayRef): ATTensor {.importcpp: "#.reshape({@})".}
func view*(self: ATTensor, size: IntArrayRef): ATTensor {.importcpp: "#.reshape({@})".}

# Automatic Differentiation
# -----------------------------------------------------------------------

func backward*(self: var ATTensor){.importcpp: "#.backward()".}

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

func `not`*(self: ATTensor): ATTensor {.importcpp: "(~#)".}
func `-`*(self: ATTensor): ATTensor {.importcpp: "(-#)".}

func `+`*(self: ATTensor, b: ATTensor): ATTensor {.importcpp: "(# + #)".}
func `-`*(self: ATTensor, b: ATTensor): ATTensor {.importcpp: "(# - #)".}
func `*`*(self: ATTensor, b: ATTensor): ATTensor {.importcpp: "(# * #)".}

func `*`*(a: cfloat or cdouble, b: ATTensor): ATTensor {.importcpp: "(# * #)".}
func `*`*(self: ATTensor, b: cfloat or cdouble): ATTensor {.importcpp: "(# * #)".}

func `+=`*(self: var ATTensor, b: ATTensor) {.importcpp: "(# += #)".}
func `+=`*(self: var ATTensor, s: Scalar) {.importcpp: "(# += #)".}
func `-=`*(self: var ATTensor, b: ATTensor) {.importcpp: "(# -= #)".}
func `-=`*(self: var ATTensor, s: Scalar) {.importcpp: "(# -= #)".}
func `*=`*(self: var ATTensor, b: ATTensor) {.importcpp: "(# *= #)".}
func `*=`*(self: var ATTensor, s: Scalar) {.importcpp: "(# *= #)".}
func `/=`*(self: var ATTensor, b: ATTensor) {.importcpp: "(# /= #)".}
func `/=`*(self: var ATTensor, s: Scalar) {.importcpp: "(# /= #)".}

func `and`*(self: ATTensor, b: ATTensor): ATTensor {.importcpp: "#.bitwise_and(#)".}
  ## bitwise `and`.
func `or`*(self: ATTensor, b: ATTensor): ATTensor {.importcpp: "#.bitwise_or(#)".}
  ## bitwise `or`.
func `xor`*(self: ATTensor, b: ATTensor): ATTensor {.importcpp: "#.bitwise_xor(#)".}
  ## bitwise `xor`.

func bitand_mut*(self: var ATTensor, s: ATTensor) {.importcpp: "#.bitwise_and_(#)".}
  ## In-place bitwise `and`.
func bitor_mut*(self: var ATTensor, s: ATTensor) {.importcpp: "#.bitwise_or_(#)".}
  ## In-place bitwise `or`.
func bitxor_mut*(self: var ATTensor, s: ATTensor) {.importcpp: "#.bitwise_xor_(#)".}
  ## In-place bitwise `xor`.

func eq*(a, b: ATTensor): ATTensor {.importcpp: "#.eq(#)".}
  ## Equality of each tensor values
func equal*(a, b: ATTensor): bool {.importcpp: "#.equal(#)".}
template `==`*(a, b: ATTensor): bool =
  a.equal(b)

# Functions.h
# -----------------------------------------------------------------------

func toType*(self: ATTensor, dtype: ScalarKind): ATTensor {.importcpp: "#.toType(@)".}
func toSparse*(self: ATTensor): ATTensor {.importcpp: "#.to_sparse()".}
func toSparse*(self: ATTensor, sparseDim: int64): ATTensor {.importcpp: "#.to_sparse(@)".}

func eye*(n: int64): ATTensor {.importcpp: "torch::eye(@)".}
func eye*(n: int64, options: TensorOptions): ATTensor {.importcpp: "torch::eye(@)".}
func eye*(n: int64, scalarKind: ScalarKind): ATTensor {.importcpp: "torch::eye(@)".}
func eye*(n: int64, device: DeviceKind): ATTensor {.importcpp: "torch::eye(@)".}

func zeros*(dim: int64): ATTensor {.importcpp: "torch::zeros(@)".}
func zeros*(dim: IntArrayRef): ATTensor {.importcpp: "torch::zeros(@)".}
func zeros*(dim: IntArrayRef, options: TensorOptions): ATTensor {.importcpp: "torch::zeros(@)".}
func zeros*(dim: IntArrayRef, scalarKind: ScalarKind): ATTensor {.importcpp: "torch::zeros(@)".}
func zeros*(dim: IntArrayRef, device: DeviceKind): ATTensor {.importcpp: "torch::zeros(@)".}

func linspace*(start, stop: Scalar, steps: int64, options: TensorOptions): ATTensor {.importcpp: "torch::linspace(@)".}
func linspace*(start, stop: Scalar, steps: int64, options: ScalarKind): ATTensor {.importcpp: "torch::linspace(@)".}
func linspace*(start, stop: Scalar, steps: int64, options: DeviceKind): ATTensor {.importcpp: "torch::linspace(@)".}
func linspace*(start, stop: Scalar, steps: int64, options: Device): ATTensor {.importcpp: "torch::linspace(@)".}
func linspace*(start, stop: Scalar, steps: int64): ATTensor {.importcpp: "torch::linspace(@)".}
func linspace*(start, stop: Scalar): ATTensor {.importcpp: "torch::linspace(@)".}

func logspace*(start, stop: Scalar, steps, base: int64, options: TensorOptions): ATTensor {.importcpp: "torch::logspace(@)".}
func logspace*(start, stop: Scalar, steps, base: int64, options: ScalarKind): ATTensor {.importcpp: "torch::logspace(@)".}
func logspace*(start, stop: Scalar, steps, base: int64, options: DeviceKind) {.importcpp: "torch::logspace(@)".}
func logspace*(start, stop: Scalar, steps, base: int64, options: Device): ATTensor {.importcpp: "torch::logspace(@)".}
func logspace*(start, stop: Scalar, steps, base: int64): ATTensor {.importcpp: "torch::logspace(@)".}
func logspace*(start, stop: Scalar, steps: int64): ATTensor {.importcpp: "torch::logspace(@)".}
func logspace*(start, stop: Scalar): ATTensor {.importcpp: "torch::logspace(@)".}

func arange*(stop: Scalar, options: TensorOptions): ATTensor {.importcpp: "torch::arange(@)".}
func arange*(stop: Scalar, options: ScalarKind): ATTensor {.importcpp: "torch::arange(@)".}
func arange*(stop: Scalar, options: DeviceKind): ATTensor {.importcpp: "torch::arange(@)".}
func arange*(stop: Scalar, options: Device): ATTensor {.importcpp: "torch::arange(@)".}
func arange*(stop: Scalar): ATTensor {.importcpp: "torch::arange(@)".}

func arange*(start, stop: Scalar, options: TensorOptions): ATTensor {.importcpp: "torch::arange(@)".}
func arange*(start, stop: Scalar, options: ScalarKind): ATTensor {.importcpp: "torch::arange(@)".}
func arange*(start, stop: Scalar, options: DeviceKind): ATTensor {.importcpp: "torch::arange(@)".}
func arange*(start, stop: Scalar, options: Device): ATTensor {.importcpp: "torch::arange(@)".}
func arange*(start, stop: Scalar): ATTensor {.importcpp: "torch::arange(@)".}

func arange*(start, stop, step: Scalar, options: TensorOptions): ATTensor {.importcpp: "torch::arange(@)".}
func arange*(start, stop, step: Scalar, options: ScalarKind): ATTensor {.importcpp: "torch::arange(@)".}
func arange*(start, stop, step: Scalar, options: DeviceKind): ATTensor {.importcpp: "torch::arange(@)".}
func arange*(start, stop, step: Scalar, options: Device): ATTensor {.importcpp: "torch::arange(@)".}
func arange*(start, stop, step: Scalar): ATTensor {.importcpp: "torch::arange(@)".}

# Operations
# -----------------------------------------------------------------------
func add*(self: ATTensor, other: ATTensor, alpha: Scalar = 1): ATTensor {.importcpp: "#.add(@)".}
func add*(self: ATTensor, other: Scalar, alpha: Scalar = 1): ATTensor {.importcpp: "#.add(@)".}
func addmv*(self: ATTensor, mat: ATTensor, vec: ATTensor, beta: Scalar = 1, alpha: Scalar = 1): ATTensor {.importcpp: "#.addmv(@)".}
func addmm*(t, mat1, mat2: ATTensor, beta: Scalar = 1, alpha: Scalar = 1): ATTensor {.importcpp: "#.addmm(@)".}
func mm*(t, other: ATTensor): ATTensor {.importcpp: "#.mm(@)".}
func matmul*(t, other: ATTensor): ATTensor {.importcpp: "#.matmul(@)".}
func bmm*(t, other: ATTensor): ATTensor {.importcpp: "#.bmm(@)".}

func luSolve*(t, data, pivots: ATTensor): ATTensor {.importcpp: "#.lu_solve(@)".}

func qr*(self: ATTensor, some: bool = true): CppTuple2[ATTensor, ATTensor] {.importcpp: "#.qr(@)".}
  ## Returns a tuple:
  ## - Q of shape (∗,m,k)
  ## - R of shape (∗,k,n)
  ## with k=min(m,n) if some is true otherwise k=m
  ##
  ## The QR decomposition is batched over dimension(s) *
  ## t = QR

# addr?
func all*(self: ATTensor, axis: int64): ATTensor {.importcpp: "#.all(@)".}
func all*(self: ATTensor, axis: int64, keepdim: bool): ATTensor {.importcpp: "#.all(@)".}
func allClose*(t, other: ATTensor, rtol: float64 = 1e-5, abstol: float64 = 1e-8, equalNan: bool = false): bool {.importcpp: "#.allclose(@)".}
func any*(self: ATTensor, axis: int64): ATTensor {.importcpp: "#.any(@)".}
func any*(self: ATTensor, axis: int64, keepdim: bool): ATTensor {.importcpp: "#.any(@)".}
func argmax*(self: ATTensor): ATTensor {.importcpp: "#.argmax()".}
func argmax*(self: ATTensor, axis: int64, keepdim: bool = false): ATTensor {.importcpp: "#.argmax(@)".}
func argmin*(self: ATTensor): ATTensor {.importcpp: "#.argmin()".}
func argmin*(self: ATTensor, axis: int64, keepdim: bool = false): ATTensor {.importcpp: "#.argmin(@)".}

# aggregate
# -----------------------------------------------------------------------

# sum needs wrapper procs/templates to allow for using nim arrays and single axis.
func sum*(self: ATTensor): ATTensor {.importcpp: "#.sum()".}
func sum*(self: ATTensor, dtype: ScalarKind): ATTensor {.importcpp: "#.sum(@)".}
func sum*(self: ATTensor, axis: int64, keepdim: bool = false): ATTensor {.importcpp: "#.sum(@)".}
func sum*(self: ATTensor, axis: int64, keepdim: bool = false, dtype: ScalarKind): ATTensor {.importcpp: "#.sum(@)".}
func sum*(self: ATTensor, axis: IntArrayRef, keepdim: bool = false): ATTensor {.importcpp: "#.sum(@)".}
func sum*(self: ATTensor, axis: IntArrayRef, keepdim: bool = false, dtype: ScalarKind): ATTensor {.importcpp: "#.sum(@)".}

# mean as well
func mean*(self: ATTensor): ATTensor {.importcpp: "#.mean()".}
func mean*(self: ATTensor, dtype: ScalarKind): ATTensor {.importcpp: "#.mean(@)".}
func mean*(self: ATTensor, axis: int64, keepdim: bool = false): ATTensor {.importcpp: "#.mean(@)".}
func mean*(self: ATTensor, axis: int64, keepdim: bool = false, dtype: ScalarKind): ATTensor {.importcpp: "#.mean(@)".}
func mean*(self: ATTensor, axis: IntArrayRef, keepdim: bool = false): ATTensor {.importcpp: "#.mean(@)".}
func mean*(self: ATTensor, axis: IntArrayRef, keepdim: bool = false, dtype: ScalarKind): ATTensor {.importcpp: "#.mean(@)".}

# median requires std::tuple

func prod*(self: ATTensor): ATTensor {.importcpp: "#.prod()".}
func prod*(self: ATTensor, dtype: ScalarKind): ATTensor {.importcpp: "#.prod(@)".}
func prod*(self: ATTensor, axis: int64, keepdim: bool = false): ATTensor {.importcpp: "#.prod(@)".}
func prod*(self: ATTensor, axis: int64, keepdim: bool = false, dtype: ScalarKind): ATTensor {.importcpp: "#.prod(@)".}

func min*(self: ATTensor): ATTensor {.importcpp: "#.min()".}
func min*(self: ATTensor, axis: int64, keepdim: bool = false): CppTuple2[ATTensor, ATTensor] {.importcpp: "torch::min(@)".}
  ## Returns a tuple (values, indices) of type (TensorT, TensorInt64)
  ## of the minimum values and their index in the specified axis

func max*(self: ATTensor): ATTensor {.importcpp: "#.max()".}
func max*(self: ATTensor, axis: int64, keepdim: bool = false): CppTuple2[ATTensor, ATTensor] {.importcpp: "torch::max(@)".}
  ## Returns a tuple (values, indices) of type (TensorT, TensorInt64)
  ## of the maximum values and their index in the specified axis

func variance*(self: ATTensor, unbiased: bool = true): ATTensor {.importcpp: "#.var(@)".} # can't use `var` because of keyword.
func variance*(self: ATTensor, axis: int64, unbiased: bool = true, keepdim: bool = false): ATTensor {.importcpp: "#.var(@)".}
func variance*(self: ATTensor, axis: IntArrayRef, unbiased: bool = true, keepdim: bool = false): ATTensor {.importcpp: "#.var(@)".}

func stddev*(self: ATTensor, unbiased: bool = true): ATTensor {.importcpp: "#.std(@)".}
func stddev*(self: ATTensor, axis: int64, unbiased: bool = true, keepdim: bool = false): ATTensor {.importcpp: "#.std(@)".}
func stddev*(self: ATTensor, axis: IntArrayRef, unbiased: bool = true, keepdim: bool = false): ATTensor {.importcpp: "#.std(@)".}

# algorithms:
# -----------------------------------------------------------------------

func sort*(self: ATTensor, axis: int64 = -1, descending: bool = false): CppTuple2[ATTensor, ATTensor] {.importcpp: "#.sort(@)".}
  ## Sorts the elements of the input tensor along a given dimension in ascending order by value.
  ## If dim is not given, the last dimension of the input is chosen (dim=-1).
  ## Returns (values, originalIndices) or type (TensorT, TensorInt64)
  ## where originalIndices is the original index of each values (before sorting)
func argsort*(self: ATTensor, axis: int64 = -1, descending: bool = false): ATTensor {.importcpp: "#.argsort(@)".}

# math
# -----------------------------------------------------------------------
func abs*(self: ATTensor): ATTensor {.importcpp: "#.abs()".}
func absolute*(self: ATTensor): ATTensor {.importcpp: "#.absolute()".}
func angle*(self: ATTensor): ATTensor {.importcpp: "#.angle()".}
func sgn*(self: ATTensor): ATTensor {.importcpp: "#.sgn()".}
func conj*(self: ATTensor): ATTensor {.importcpp: "#.conj()".}
func acos*(self: ATTensor): ATTensor {.importcpp: "#.acos()".}
func arccos*(self: ATTensor): ATTensor {.importcpp: "#.arccos()".}
func acosh*(self: ATTensor): ATTensor {.importcpp: "#.acosh()".}
func arccosh*(self: ATTensor): ATTensor {.importcpp: "#.arccosh()".}
func asinh*(self: ATTensor): ATTensor {.importcpp: "#.asinh()".}
func arcsinh*(self: ATTensor): ATTensor {.importcpp: "#.arcsinh()".}
func atanh*(self: ATTensor): ATTensor {.importcpp: "#.atanh()".}
func arctanh*(self: ATTensor): ATTensor {.importcpp: "#.arctanh()".}
func asin*(self: ATTensor): ATTensor {.importcpp: "#.asin()".}
func arcsin*(self: ATTensor): ATTensor {.importcpp: "#.arcsin()".}
func atan*(self: ATTensor): ATTensor {.importcpp: "#.atan()".}
func arctan*(self: ATTensor): ATTensor {.importcpp: "#.arctan()".}
func cos*(self: ATTensor): ATTensor {.importcpp: "#.cos()".}
func sin*(self: ATTensor): ATTensor {.importcpp: "#.sin()".}
func tan*(self: ATTensor): ATTensor {.importcpp: "#.tan()".}
func exp*(self: ATTensor): ATTensor {.importcpp: "#.exp()".}
func exp2*(self: ATTensor): ATTensor {.importcpp: "#.exp2()".}
func erf*(self: ATTensor): ATTensor {.importcpp: "#.erf()".}
func erfc*(self: ATTensor): ATTensor {.importcpp: "#.erfc()".}
func reciprocal*(self: ATTensor): ATTensor {.importcpp: "#.reciprocal()".}
func neg*(self: ATTensor): ATTensor {.importcpp: "#.neg()".}
func clamp*(self: ATTensor, min, max: Scalar): ATTensor {.importcpp: "#.clamp(@)".}
func clampMin*(self: ATTensor, min: Scalar): ATTensor {.importcpp: "#.clamp_min(@)".}
func clampMax*(self: ATTensor, max: Scalar): ATTensor {.importcpp: "#.clamp_max(@)".}

func dot*(self: ATTensor, other: ATTensor): ATTensor {.importcpp: "#.dot(@)".}

func squeeze*(self: ATTensor): ATTensor {.importcpp: "#.squeeze()".}
func squeeze*(self: ATTensor, axis: int64): ATTensor {.importcpp: "#.squeeze(@)".}
func unsqueeze*(self: ATTensor, axis: int64): ATTensor {.importcpp: "#.unsqueeze(@)".}

# FFT
# -----------------------------------------------------------------------
func fftshift*(self: ATTensor): ATTensor {.importcpp: "torch::fft_fftshift(@)".}
func fftshift*(self: ATTensor, dim: IntArrayRef): ATTensor {.importcpp: "torch::fft_ifftshift(@)".}
func ifftshift*(self: ATTensor): ATTensor {.importcpp: "torch::fft_fftshift(@)".}
func ifftshift*(self: ATTensor, dim: IntArrayRef): ATTensor {.importcpp: "torch::fft_ifftshift(@)".}

let defaultNorm: CppString = initCppString("backward")

func fft*(self: ATTensor, n: int64, dim: int64 = -1, norm: CppString = defaultNorm): ATTensor {.importcpp: "torch::fft_fft(@)".}
## Compute the 1-D Fourier transform
## ``n`` represent Signal length. If given, the input will either be zero-padded or trimmed to this length before computing the FFT.
## ``norm`` can be :
##    * "forward" - normalize by 1/n
##    * "backward" - no normalization
##    * "ortho" - normalize by 1/sqrt(n)
func fft*(self: ATTensor): ATTensor {.importcpp: "torch::fft_fft(@)".}
## Compute the 1-D Fourier transform

func ifft*(self: ATTensor, n: int64, dim: int64 = -1, norm: CppString = defaultNorm): ATTensor {.importcpp: "torch::fft_ifft(@)".}
## Compute the 1-D Fourier transform
## ``n`` represent Signal length. If given, the input will either be zero-padded or trimmed to this length before computing the FFT.
## ``norm`` can be :
##   * "forward" - no normalization
##   * "backward" - normalization by 1/n
##   * "ortho" - normalization by 1/sqrt(n)
func ifft*(self: ATTensor): ATTensor {.importcpp: "torch::fft_ifft(@)".}
## Compute the 1-D Fourier transform

func fft2*(self: ATTensor, s: IntArrayRef, dim: IntArrayRef, norm: CppString = defaultNorm): ATTensor {.importcpp: "torch::fft_fft2(@)".}
## Compute the 2-D Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
## ``norm`` can be :
##    * "forward" - normalize by 1/n
##    * "backward" - no normalization
##    * "ortho" - normalize by 1/sqrt(n)
## With n the logical FFT size: ``n = prod(s)``.
func fft2*(self: ATTensor, s: IntArrayRef): ATTensor {.importcpp: "torch::fft_fft2(@)".}
## Compute the 2-D Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
func fft2*(self: ATTensor): ATTensor {.importcpp: "torch::fft_fft2(@)".}
## Compute the 2-D Fourier transform

func ifft2*(self: ATTensor, s: IntArrayRef, dim: IntArrayRef, norm: CppString = defaultNorm): ATTensor {.importcpp: "torch::fft_ifft2(@)".}
## Compute the 2-D Inverse Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
## ``norm`` can be :
##   * "forward" - no normalization
##   * "backward" - normalization by 1/n
##   * "ortho" - normalization by 1/sqrt(n)
## With n the logical FFT size: ``n = prod(s)``.
func ifft2*(self: ATTensor, s: IntArrayRef): ATTensor {.importcpp: "torch::fft_ifft2(@)".}
## Compute the 2-D Inverse Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
func ifft2*(self: ATTensor): ATTensor {.importcpp: "torch::fft_ifft2(@)".}
## Compute the 2-D Inverse Fourier transform

func fftn*(self: ATTensor, s: IntArrayRef, dim: IntArrayRef, norm: CppString = defaultNorm): ATTensor {.importcpp: "torch::fft_fftn(@)".}
## Compute the N-D Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
## ``norm`` can be :
##    * "forward" normalize by 1/n
##    * "backward" - no normalization
##    * "ortho" normalize by 1/sqrt(n)
## With n the logical FFT size: ``n = prod(s)``.
func fftn*(self: ATTensor, s: IntArrayRef): ATTensor {.importcpp: "torch::fft_fftn(@)".}
## Compute the N-D Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
func fftn*(self: ATTensor): ATTensor {.importcpp: "torch::fft_fftn(@)".}
## Compute the N-D Fourier transform

func ifftn*(self: ATTensor, s: IntArrayRef, dim: IntArrayRef, norm: CppString = defaultNorm): ATTensor {.importcpp: "torch::fft_ifftn(@)".}
## Compute the N-D Inverse Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
## ``norm`` can be :
##   * "forward" - no normalization
##   * "backward" - normalization by 1/n
##   * "ortho" - normalization by 1/sqrt(n)
## With n the logical FFT size: ``n = prod(s)``.
func ifftn*(self: ATTensor, s: IntArrayRef): ATTensor {.importcpp: "torch::fft_ifftn(@)".}
## Compute the N-D Inverse Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
func ifftn*(self: ATTensor): ATTensor {.importcpp: "torch::fft_ifftn(@)".}
## Compute the N-D Inverse Fourier transform

func rfft*(self: ATTensor, n: int64, dim: int64 = -1, norm: CppString = defaultNorm): ATTensor {.importcpp: "torch::fft_rfft(@)".}
## Computes the one dimensional Fourier transform of real-valued input.
func rfft*(self: ATTensor): ATTensor {.importcpp: "torch::fft_rfft(@)".}
## Computes the one dimensional Fourier transform of real-valued input.

func irfft*(self: ATTensor, n: int64, dim: int64 = -1, norm: CppString = defaultNorm): ATTensor {.importcpp: "torch::fft_irfft(@)".}
## Computes the one dimensional Fourier transform of real-valued input.
func irfft*(self: ATTensor): ATTensor {.importcpp: "torch::fft_irfft(@)".}
## Computes the one dimensional Fourier transform of real-valued input.

func rfft2*(self: ATTensor, s: IntArrayRef, dim: IntArrayRef, norm: CppString = defaultNorm): ATTensor {.importcpp: "torch::fft_rfft2(@)".}
## Compute the N-D Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
## ``norm`` can be :
##    * "forward" - normalize by 1/n
##    * "backward" - no normalization
##    * "ortho" - normalize by 1/sqrt(n)
## With n the logical FFT size: ``n = prod(s)``.
func rfft2*(self: ATTensor, s: IntArrayRef): ATTensor {.importcpp: "torch::fft_rfft2(@)".}
## Compute the N-D Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
func rfft2*(self: ATTensor): ATTensor {.importcpp: "torch::fft_rfft2(@)".}
## Compute the N-D Fourier transform

func irfft2*(self: ATTensor, s: IntArrayRef, dim: IntArrayRef, norm: CppString = defaultNorm): ATTensor {.importcpp: "torch::fft_irfft2(@)".}
## Compute the N-D Inverse Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
## ``norm`` can be :
##   * "forward" - no normalization
##   * "backward" - normalization by 1/n
##   * "ortho" - normalization by 1/sqrt(n)
## With n the logical FFT size: ``n = prod(s)``.
func irfft2*(self: ATTensor, s: IntArrayRef): ATTensor {.importcpp: "torch::fft_irfft2(@)".}
## Compute the N-D Inverse Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
func irfft2*(self: ATTensor): ATTensor {.importcpp: "torch::fft_irfft2(@)".}
## Compute the N-D Inverse Fourier transform


func rfftn*(self: ATTensor, s: IntArrayRef, dim: IntArrayRef, norm: CppString = defaultNorm): ATTensor {.importcpp: "torch::fft_rfftn(@)".}
## Compute the N-D Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
## ``norm`` can be :
##    * "forward" - normalize by 1/n
##    * "backward" - no normalization
##    * "ortho" - normalize by 1/sqrt(n)
## With n the logical FFT size: ``n = prod(s)``.
func rfftn*(self: ATTensor, s: IntArrayRef): ATTensor {.importcpp: "torch::fft_rfftn(@)".}
## Compute the N-D Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
func rfftn*(self: ATTensor): ATTensor {.importcpp: "torch::fft_rfftn(@)".}
## Compute the N-D Fourier transform

func irfftn*(self: ATTensor, s: IntArrayRef, dim: IntArrayRef, norm: CppString = defaultNorm): ATTensor {.importcpp: "torch::fft_irfftn(@)".}
## Compute the N-D Inverse Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
## ``norm`` can be :
##   * "forward" - no normalization
##   * "backward" - normalization by 1/n
##   * "ortho" - normalization by 1/sqrt(n)
## With n the logical FFT size: ``n = prod(s)``.
func irfftn*(self: ATTensor, s: IntArrayRef): ATTensor {.importcpp: "torch::fft_irfftn(@)".}
## Compute the N-D Inverse Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
func irfftn*(self: ATTensor): ATTensor {.importcpp: "torch::fft_irfftn(@)".}
## Compute the N-D Inverse Fourier transform

func hfft*(self: ATTensor, n: int64, dim: int64 = -1, norm: CppString = defaultNorm): ATTensor {.importcpp: "torch::hfft(@)".}
## Computes the 1 dimensional FFT of a onesided Hermitian signal.
func hfft*(self: ATTensor): ATTensor {.importcpp: "torch::hfft(@)".}
## Computes the 1 dimensional FFT of a onesided Hermitian signal.
func ihfft*(self: ATTensor, n: int64, dim: int64 = -1, norm: CppString = defaultNorm): ATTensor {.importcpp: "torch::ihfft(@)".}
## Computes the inverse FFT of a real-valued Fourier domain signal.
func ihfft*(self: ATTensor): ATTensor {.importcpp: "torch::ihfft(@)".}
## Computes the inverse FFT of a real-valued Fourier domain signal.
{.pop.}
#func convolution*(self: Tensor, weight: Tensor, bias: Tensor, stride, padding, dilation: int64, transposed: bool, outputPadding: int64, groups: int64): Tensor {.importcpp: "torch::convolution(@)".}
