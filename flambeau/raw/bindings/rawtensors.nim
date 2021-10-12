# Flambeau
# Copyright (c) 2020 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  # Standard library
  std/complex,
  cppstl/std_string,
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
type Scalar* = SomeNumber or bool or C10_Complex

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
  RawTensor* {.importcpp: "torch::Tensor", bycopy.} = object

# Strings & Debugging
# -----------------------------------------------------------------------

proc print*(self: RawTensor) {.sideeffect, importcpp: "torch::print(@)".}

# Metadata
# -----------------------------------------------------------------------

func dim*(self: RawTensor): int64 {.importcpp: "#.dim()".}
  ## Number of dimensions
func reset*(self: var RawTensor) {.importcpp: "#.reset()".}
func is_same*(self, other: RawTensor): bool {.importcpp: "#.is_same(#)".}
  ## Reference equality
  ## Do the tensors use the same memory.

func sizes*(self: RawTensor): IntArrayRef {.importcpp: "#.sizes()".}
  ## This is Arraymancer and Numpy "shape"

func strides*(self: RawTensor): IntArrayRef {.importcpp: "#.strides()".}

func ndimension*(self: RawTensor): int64 {.importcpp: "#.ndimension()".}
  ## This is Arraymancer rank
func nbytes*(self: RawTensor): uint {.importcpp: "#.nbytes()".}
  ## Bytes-size of the Tensor
func numel*(self: RawTensor): int64 {.importcpp: "#.numel()".}
  ## This is Arraymancer and Numpy "size"

func size*(self: RawTensor, axis: int64): int64 {.importcpp: "#.size(#)".}
func itemsize*(self: RawTensor): uint {.importcpp: "#.itemsize()".}
func element_size*(self: RawTensor): int64 {.importcpp: "#.element_size()".}

# Accessors
# -----------------------------------------------------------------------

func data_ptr*(self: RawTensor, T: typedesc): ptr UncheckedArray[T] {.importcpp: "#.data_ptr<'2>(#)".}
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

func has_storage*(self: RawTensor): bool {.importcpp: "#.has_storage()".}
func get_device*(self: RawTensor): int64 {.importcpp: "#.get_device()".}
func is_cuda*(self: RawTensor): bool {.importcpp: "#.is_cuda()".}
func is_hip*(self: RawTensor): bool {.importcpp: "#.is_hip()".}
func is_sparse*(self: RawTensor): bool {.importcpp: "#.is_sparse()".}
func is_mkldnn*(self: RawTensor): bool {.importcpp: "#.is_mkldnn()".}
func is_vulkan*(self: RawTensor): bool {.importcpp: "#.is_vulkan()".}
func is_quantized*(self: RawTensor): bool {.importcpp: "#.is_quantized()".}
func is_meta*(self: RawTensor): bool {.importcpp: "#.is_meta()".}

func cpu*(self: RawTensor): RawTensor {.importcpp: "#.cpu()".}
func cuda*(self: RawTensor): RawTensor {.importcpp: "#.cuda()".}
func hip*(self: RawTensor): RawTensor {.importcpp: "#.hip()".}
func vulkan*(self: RawTensor): RawTensor {.importcpp: "#.vulkan()".}
func to*(self: RawTensor, device: DeviceKind): RawTensor {.importcpp: "#.to(#)".}
func to*(self: RawTensor, device: Device): RawTensor {.importcpp: "#.to(#)".}

# dtype
# -----------------------------------------------------------------------

func to*(self: RawTensor, dtype: ScalarKind): RawTensor {.importcpp: "#.to(#)".}
func scalarType*(self: RawTensor): ScalarKind {.importcpp: "#.scalar_type()".}

# Constructors
# -----------------------------------------------------------------------

# DeviceType and ScalarType are auto-convertible to TensorOptions

func init*(T: type RawTensor): RawTensor {.constructor, importcpp: "torch::Tensor".}
# Default empty constructor
func initRawTensor*(): RawTensor {.constructor, importcpp: "torch::Tensor".}
# Move / Copy constructor ?
func initRawTensor*(t: RawTensor): RawTensor {.constructor, importcpp: "torch::Tensor(@)".}

func from_blob*(data: pointer, sizes: IntArrayRef, options: TensorOptions): RawTensor {.importcpp: "torch::from_blob(@)".}
func from_blob*(data: pointer, sizes: IntArrayRef, scalarKind: ScalarKind): RawTensor {.importcpp: "torch::from_blob(@)".}
func from_blob*(data: pointer, sizes: IntArrayRef, device: DeviceKind): RawTensor {.importcpp: "torch::from_blob(@)".}

func from_blob*(data: pointer, sizes: int64, options: TensorOptions): RawTensor {.importcpp: "torch::from_blob(@)".}
func from_blob*(data: pointer, sizes: int64, scalarKind: ScalarKind): RawTensor {.importcpp: "torch::from_blob(@)".}
func from_blob*(data: pointer, sizes: int64, device: DeviceKind): RawTensor {.importcpp: "torch::from_blob(@)".}

func from_blob*(data: pointer, sizes, strides: IntArrayRef, options: TensorOptions): RawTensor {.importcpp: "torch::from_blob(@)".}
func from_blob*(data: pointer, sizes, strides: IntArrayRef, scalarKind: ScalarKind): RawTensor {.importcpp: "torch::from_blob(@)".}
func from_blob*(data: pointer, sizes, strides: IntArrayRef, device: DeviceKind): RawTensor {.importcpp: "torch::from_blob(@)".}

func empty*(size: IntArrayRef, options: TensorOptions): RawTensor {.importcpp: "torch::empty(@)".}
  ## Create an uninitialized tensor of shape `size`
  ## The tensor data must be filled manually
  ##
  ## The output tensor will be row major (C contiguous)
func empty*(size: IntArrayRef, scalarKind: ScalarKind): RawTensor {.importcpp: "torch::empty(@)".}
func empty*(size: IntArrayRef, device: DeviceKind): RawTensor {.importcpp: "torch::empty(@)".}
  ## Create an uninitialized tensor of shape `size`
  ## The tensor data must be filled manually.
  ##
  ## If device is NOT on CPU make sure to use specialized
  ## copy operations. For example to update on Cuda devices
  ## use cudaMemcpy not a.data[i] = 123
  ##
  ## The output tensor will be row major (C contiguous)

func clone*(self: RawTensor): RawTensor {.importcpp: "#.clone()".}

# TODO : Test this
func view_as_real*(self: RawTensor) : RawTensor {.importcpp: "#.view_as_real()".}
func view_as_complex*(self: RawTensor) : RawTensor {.importcpp: "#.view_as_complex()".}

# Random sampling
# -----------------------------------------------------------------------
func random_mut*(self: var RawTensor, start, stopEx: int64) {.importcpp: "#.random_(@)".}
func randint*(start, stopEx: int64): RawTensor {.varargs, importcpp: "torch::randint(#, #, {@})".}
func randint*(start, stopEx: int64, size: IntArrayRef): RawTensor {.importcpp: "torch::randint(@)".}

func rand_like*(self: RawTensor, options: TensorOptions): RawTensor {.importcpp: "torch::rand_like(@)".}
func rand_like*(self: RawTensor, options: ScalarKind): RawTensor {.importcpp: "torch::rand_like(@)".}
func rand_like*(self: RawTensor, options: DeviceKind): RawTensor {.importcpp: "torch::rand_like(@)".}
func rand_like*(self: RawTensor, options: Device): RawTensor {.importcpp: "torch::rand_like(@)".}
func rand_like*(self: RawTensor): RawTensor {.importcpp: "torch::rand_like(@)".}

func rand*(size: IntArrayRef, options: TensorOptions): RawTensor {.importcpp: "torch::rand(@)".}
func rand*(size: IntArrayRef, options: DeviceKind): RawTensor {.importcpp: "torch::rand(@)".}
func rand*(size: IntArrayRef, options: Device): RawTensor {.importcpp: "torch::rand(@)".}
func rand*(size: IntArrayRef, options: ScalarKind): RawTensor {.importcpp: "torch::rand(@)".}
func rand*(size: IntArrayRef): RawTensor {.importcpp: "torch::rand(@)".}

# Indexing
# -----------------------------------------------------------------------
# TODO -> separate the FFI from the Nim Raw API to add IndexDefect when compileOptions("boundsCheck")
# libtorch/include/ATen/TensorIndexing.h
# and https://pytorch.org/cppdocs/notes/tensor_indexing.html

func item*(self: RawTensor, T: typedesc): T {.importcpp: "#.item<'0>()".}
  ## Extract the scalar from a 0-dimensional tensor
func item*(self: RawTensor, T: typedesc[Complex32]): C10_Complex[float32] {.importcpp: "#.item<c10::complex<float>>()".}
func item*(self: RawTensor, T: typedesc[Complex64]): C10_Complex[float64] {.importcpp: "#.item<c10::complex<double>>()".}

# Unsure what those corresponds to in Python
# func `[]`*(self: Tensor, index: Scalar): Tensor {.importcpp: "#[#]".}
# func `[]`*(self: Tensor, index: Tensor): Tensor {.importcpp: "#[#]".}
# func `[]`*(self: Tensor, index: int64): Tensor {.importcpp: "#[#]".}

func index*(self: RawTensor): RawTensor {.varargs, importcpp: "#.index({@})".}
  ## Tensor indexing. It is recommended
  ## to Nimify this in a high-level wrapper.
  ## `tensor.index(indexers)`

# We can't use the construct `#.index_put_({@}, #)`
# so hardcode sizes,
# 6d seems reasonable, that would be a batch of 3D videos (videoID/batchID, Time, Color Channel, Height, Width, Depth)
# If you need more you likely aren't indexing individual values.

func index_put*(self: var RawTensor, i0: auto, val: Scalar or RawTensor) {.importcpp: "#.index_put_({#}, #)".}
  ## Tensor mutation at index. It is recommended
  ## to Nimify this in a high-level wrapper.
func index_put*(self: var RawTensor, i0, i1: auto, val: Scalar or RawTensor) {.importcpp: "#.index_put_({#, #}, #)".}
  ## Tensor mutation at index. It is recommended
  ## to Nimify this in a high-level wrapper.
func index_put*(self: var RawTensor, i0, i1, i2: auto, val: Scalar or RawTensor) {.importcpp: "#.index_put_({#, #, #}, #)".}
  ## Tensor mutation at index. It is recommended
  ## to Nimify this in a high-level wrapper.
func index_put*(self: var RawTensor, i0, i1, i2, i3: auto, val: Scalar or RawTensor) {.importcpp: "#.index_put_({#, #, #, #}, #)".}
  ## Tensor mutation at index. It is recommended
  ## to Nimify this in a high-level wrapper.
func index_put*(self: var RawTensor, i0, i1, i2, i3, i4: auto, val: Scalar or RawTensor) {.importcpp: "#.index_put_({#, #, #, #, #}, #)".}
  ## Tensor mutation at index. It is recommended
  ## to Nimify this in a high-level wrapper.
func index_put*(self: var RawTensor, i0, i1, i2, i3, i4, i5: auto, val: Scalar or RawTensor) {.importcpp: "#.index_put_({#, #, #, #, #, #}, #)".}
  ## Tensor mutation at index. It is recommended
  ## to Nimify this in a high-level wrapper.

# Fancy Indexing
# -----------------------------------------------------------------------
# TODO -> separate the FFI from the Nim Raw API to add IndexDefect when compileOptions("boundsCheck")
func index_select*(self: RawTensor, axis: int64, indices: RawTensor): RawTensor {.importcpp: "#.index_select(@)".}
func masked_select*(self: RawTensor, mask: RawTensor): RawTensor {.importcpp: "#.masked_select(@)".}

# PyTorch exposes in-place `index_fill_` and `masked_fill_`
# and out-of-place `index_fill` and `masked_fill`
# that does in-place + clone
# we only exposes the in-place version.

func index_fill_mut*(self: var RawTensor, mask: RawTensor, value: Scalar or RawTensor) {.importcpp: "#.index_fill_(@)".}
func masked_fill_mut*(self: var RawTensor, mask: RawTensor, value: Scalar or RawTensor) {.importcpp: "#.masked_fill_(@)".}

# Shapeshifting
# -----------------------------------------------------------------------

func reshape*(self: RawTensor, sizes: IntArrayRef): RawTensor {.importcpp: "#.reshape({@})".}
func view*(self: RawTensor, size: IntArrayRef): RawTensor {.importcpp: "#.reshape({@})".}

# Automatic Differentiation
# -----------------------------------------------------------------------

func backward*(self: var RawTensor) {.importcpp: "#.backward()".}

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

# The None used in Torch isn't actually the enum but a c10::nullopt
let None* {.importcpp: "torch::indexing::None".} : Nullopt_t

type EllipsisIndexType* {.importcpp: "torch::indexing::EllipsisIndexType".} = object

let Ellipsis* {.importcpp: "torch::indexing::Ellipsis".} : EllipsisIndexType

  # SomeSlicer* = TensorIndexType|SomeSignedInt

proc SliceSpan*(): TorchSlice {.importcpp: "at::indexing::Slice()".}
    ## This is passed to the "index" function
    ## This is Python ":", span / whole dimension

func torchSlice*(){.importcpp: "torch::indexing::Slice(@)", constructor.}
func torchSlice*(start: Nullopt_t|SomeSignedInt): TorchSlice {.importcpp: "torch::indexing::Slice(@)", constructor.}
func torchSlice*(start: Nullopt_t|SomeSignedInt, stop: Nullopt_t|SomeSignedInt): TorchSlice {.importcpp: "torch::indexing::Slice(@)", constructor.}
func torchSlice*(start: Nullopt_t|SomeSignedInt, stop: Nullopt_t|SomeSignedInt, step: Nullopt_t|SomeSignedInt): TorchSlice {.importcpp: "torch::indexing::Slice(@)", constructor.}


func start*(s: TorchSlice): int64 {.importcpp: "#.start()".}
func stop*(s: TorchSlice): int64 {.importcpp: "#.stop()".}
func step*(s: TorchSlice): int64 {.importcpp: "#.step()".}

# Operators
# -----------------------------------------------------------------------
func assign*(self: var RawTensor, other: RawTensor) {.importcpp: "# = #".}

func `not`*(self: RawTensor): RawTensor {.importcpp: "(~#)".}
func `-`*(self: RawTensor): RawTensor {.importcpp: "(-#)".}

func `+`*(self: RawTensor, b: RawTensor): RawTensor {.importcpp: "(# + #)".}
func `-`*(self: RawTensor, b: RawTensor): RawTensor {.importcpp: "(# - #)".}
func `*`*(self: RawTensor, b: RawTensor): RawTensor {.importcpp: "(# * #)".}

func `*`*(a: SomeNumber, b: RawTensor): RawTensor {.importcpp: "(# * #)".}
func `*`*(self: RawTensor, b: SomeNumber): RawTensor {.importcpp: "(# * #)".}

func `+`*(a: SomeNumber, b: RawTensor): RawTensor {.importcpp: "(# + #)".}
func `+`*(self: RawTensor, b: SomeNumber): RawTensor {.importcpp: "(# + #)".}

func `+=`*(self: var RawTensor, b: RawTensor) {.importcpp: "(# += #)".}
func `+=`*(self: var RawTensor, s: Scalar) {.importcpp: "(# += #)".}
func `-=`*(self: var RawTensor, b: RawTensor) {.importcpp: "(# -= #)".}
func `-=`*(self: var RawTensor, s: Scalar) {.importcpp: "(# -= #)".}
func `*=`*(self: var RawTensor, b: RawTensor) {.importcpp: "(# *= #)".}
func `*=`*(self: var RawTensor, s: Scalar) {.importcpp: "(# *= #)".}
func `/=`*(self: var RawTensor, b: RawTensor) {.importcpp: "(# /= #)".}
func `/=`*(self: var RawTensor, s: Scalar) {.importcpp: "(# /= #)".}

func `and`*(self: RawTensor, b: RawTensor): RawTensor {.importcpp: "#.bitwise_and(#)".}
  ## bitwise `and`.
func `or`*(self: RawTensor, b: RawTensor): RawTensor {.importcpp: "#.bitwise_or(#)".}
  ## bitwise `or`.
func `xor`*(self: RawTensor, b: RawTensor): RawTensor {.importcpp: "#.bitwise_xor(#)".}
  ## bitwise `xor`.

func bitand_mut*(self: var RawTensor, s: RawTensor) {.importcpp: "#.bitwise_and_(#)".}
  ## In-place bitwise `and`.
func bitor_mut*(self: var RawTensor, s: RawTensor) {.importcpp: "#.bitwise_or_(#)".}
  ## In-place bitwise `or`.
func bitxor_mut*(self: var RawTensor, s: RawTensor) {.importcpp: "#.bitwise_xor_(#)".}
  ## In-place bitwise `xor`.

func eq*(a, b: RawTensor): RawTensor {.importcpp: "#.eq(#)".}
  ## Equality of each tensor values
func equal*(a, b: RawTensor): bool {.importcpp: "#.equal(#)".}
template `==`*(a, b: RawTensor): bool =
  a.equal(b)

# Functions.h
# -----------------------------------------------------------------------

func toType*(self: RawTensor, dtype: ScalarKind): RawTensor {.importcpp: "#.toType(@)".}
func toSparse*(self: RawTensor): RawTensor {.importcpp: "#.to_sparse()".}
func toSparse*(self: RawTensor, sparseDim: int64): RawTensor {.importcpp: "#.to_sparse(@)".}

func eye*(n: int64): RawTensor {.importcpp: "torch::eye(@)".}
func eye*(n: int64, options: TensorOptions): RawTensor {.importcpp: "torch::eye(@)".}
func eye*(n: int64, scalarKind: ScalarKind): RawTensor {.importcpp: "torch::eye(@)".}
func eye*(n: int64, device: DeviceKind): RawTensor {.importcpp: "torch::eye(@)".}

func zeros*(dim: int64): RawTensor {.importcpp: "torch::zeros(@)".}
func zeros*(dim: IntArrayRef): RawTensor {.importcpp: "torch::zeros(@)".}
func zeros*(dim: IntArrayRef, options: TensorOptions): RawTensor {.importcpp: "torch::zeros(@)".}
func zeros*(dim: IntArrayRef, scalarKind: ScalarKind): RawTensor {.importcpp: "torch::zeros(@)".}
func zeros*(dim: IntArrayRef, device: DeviceKind): RawTensor {.importcpp: "torch::zeros(@)".}

func linspace*(start, stop: Scalar, steps: int64, options: TensorOptions): RawTensor {.importcpp: "torch::linspace(@)".}
func linspace*(start, stop: Scalar, steps: int64, options: ScalarKind): RawTensor {.importcpp: "torch::linspace(@)".}
func linspace*(start, stop: Scalar, steps: int64, options: DeviceKind): RawTensor {.importcpp: "torch::linspace(@)".}
func linspace*(start, stop: Scalar, steps: int64, options: Device): RawTensor {.importcpp: "torch::linspace(@)".}
func linspace*(start, stop: Scalar, steps: int64): RawTensor {.importcpp: "torch::linspace(@)".}
func linspace*(start, stop: Scalar): RawTensor {.importcpp: "torch::linspace(@)".}

func logspace*(start, stop: Scalar, steps, base: int64, options: TensorOptions): RawTensor {.importcpp: "torch::logspace(@)".}
func logspace*(start, stop: Scalar, steps, base: int64, options: ScalarKind): RawTensor {.importcpp: "torch::logspace(@)".}
func logspace*(start, stop: Scalar, steps, base: int64, options: DeviceKind) {.importcpp: "torch::logspace(@)".}
func logspace*(start, stop: Scalar, steps, base: int64, options: Device): RawTensor {.importcpp: "torch::logspace(@)".}
func logspace*(start, stop: Scalar, steps, base: int64): RawTensor {.importcpp: "torch::logspace(@)".}
func logspace*(start, stop: Scalar, steps: int64): RawTensor {.importcpp: "torch::logspace(@)".}
func logspace*(start, stop: Scalar): RawTensor {.importcpp: "torch::logspace(@)".}

func arange*(stop: Scalar, options: TensorOptions): RawTensor {.importcpp: "torch::arange(@)".}
func arange*(stop: Scalar, options: ScalarKind): RawTensor {.importcpp: "torch::arange(@)".}
func arange*(stop: Scalar, options: DeviceKind): RawTensor {.importcpp: "torch::arange(@)".}
func arange*(stop: Scalar, options: Device): RawTensor {.importcpp: "torch::arange(@)".}
func arange*(stop: Scalar): RawTensor {.importcpp: "torch::arange(@)".}

func arange*(start, stop: Scalar, options: TensorOptions): RawTensor {.importcpp: "torch::arange(@)".}
func arange*(start, stop: Scalar, options: ScalarKind): RawTensor {.importcpp: "torch::arange(@)".}
func arange*(start, stop: Scalar, options: DeviceKind): RawTensor {.importcpp: "torch::arange(@)".}
func arange*(start, stop: Scalar, options: Device): RawTensor {.importcpp: "torch::arange(@)".}
func arange*(start, stop: Scalar): RawTensor {.importcpp: "torch::arange(@)".}

func arange*(start, stop, step: Scalar, options: TensorOptions): RawTensor {.importcpp: "torch::arange(@)".}
func arange*(start, stop, step: Scalar, options: ScalarKind): RawTensor {.importcpp: "torch::arange(@)".}
func arange*(start, stop, step: Scalar, options: DeviceKind): RawTensor {.importcpp: "torch::arange(@)".}
func arange*(start, stop, step: Scalar, options: Device): RawTensor {.importcpp: "torch::arange(@)".}
func arange*(start, stop, step: Scalar): RawTensor {.importcpp: "torch::arange(@)".}

# Operations
# -----------------------------------------------------------------------
func add*(self: RawTensor, other: RawTensor, alpha: Scalar = 1): RawTensor {.importcpp: "#.add(@)".}
func add*(self: RawTensor, other: Scalar, alpha: Scalar = 1): RawTensor {.importcpp: "#.add(@)".}
func addmv*(self: RawTensor, mat: RawTensor, vec: RawTensor, beta: Scalar = 1, alpha: Scalar = 1): RawTensor {.importcpp: "#.addmv(@)".}
func addmm*(t, mat1, mat2: RawTensor, beta: Scalar = 1, alpha: Scalar = 1): RawTensor {.importcpp: "#.addmm(@)".}
func mm*(t, other: RawTensor): RawTensor {.importcpp: "#.mm(@)".}
func matmul*(t, other: RawTensor): RawTensor {.importcpp: "#.matmul(@)".}
func bmm*(t, other: RawTensor): RawTensor {.importcpp: "#.bmm(@)".}

func luSolve*(t, data, pivots: RawTensor): RawTensor {.importcpp: "#.lu_solve(@)".}

func qr*(self: RawTensor, some: bool = true): CppTuple2[RawTensor, RawTensor] {.importcpp: "#.qr(@)".}
  ## Returns a tuple:
  ## - Q of shape (∗,m,k)
  ## - R of shape (∗,k,n)
  ## with k=min(m,n) if some is true otherwise k=m
  ##
  ## The QR decomposition is batched over dimension(s) *
  ## t = QR

# addr?
func all*(self: RawTensor, axis: int64): RawTensor {.importcpp: "#.all(@)".}
func all*(self: RawTensor, axis: int64, keepdim: bool): RawTensor {.importcpp: "#.all(@)".}
func allClose*(t, other: RawTensor, rtol: float64 = 1e-5, abstol: float64 = 1e-8, equalNan: bool = false): bool {.importcpp: "#.allclose(@)".}
func any*(self: RawTensor, axis: int64): RawTensor {.importcpp: "#.any(@)".}
func any*(self: RawTensor, axis: int64, keepdim: bool): RawTensor {.importcpp: "#.any(@)".}
func argmax*(self: RawTensor): RawTensor {.importcpp: "#.argmax()".}
func argmax*(self: RawTensor, axis: int64, keepdim: bool = false): RawTensor {.importcpp: "#.argmax(@)".}
func argmin*(self: RawTensor): RawTensor {.importcpp: "#.argmin()".}
func argmin*(self: RawTensor, axis: int64, keepdim: bool = false): RawTensor {.importcpp: "#.argmin(@)".}

# aggregate
# -----------------------------------------------------------------------

# sum needs wrapper procs/templates to allow for using nim arrays and single axis.
func sum*(self: RawTensor): RawTensor {.importcpp: "#.sum()".}
func sum*(self: RawTensor, dtype: ScalarKind): RawTensor {.importcpp: "#.sum(@)".}
func sum*(self: RawTensor, axis: int64, keepdim: bool = false): RawTensor {.importcpp: "#.sum(@)".}
func sum*(self: RawTensor, axis: int64, keepdim: bool = false, dtype: ScalarKind): RawTensor {.importcpp: "#.sum(@)".}
func sum*(self: RawTensor, axis: IntArrayRef, keepdim: bool = false): RawTensor {.importcpp: "#.sum(@)".}
func sum*(self: RawTensor, axis: IntArrayRef, keepdim: bool = false, dtype: ScalarKind): RawTensor {.importcpp: "#.sum(@)".}

# mean as well
func mean*(self: RawTensor): RawTensor {.importcpp: "#.mean()".}
func mean*(self: RawTensor, dtype: ScalarKind): RawTensor {.importcpp: "#.mean(@)".}
func mean*(self: RawTensor, axis: int64, keepdim: bool = false): RawTensor {.importcpp: "#.mean(@)".}
func mean*(self: RawTensor, axis: int64, keepdim: bool = false, dtype: ScalarKind): RawTensor {.importcpp: "#.mean(@)".}
func mean*(self: RawTensor, axis: IntArrayRef, keepdim: bool = false): RawTensor {.importcpp: "#.mean(@)".}
func mean*(self: RawTensor, axis: IntArrayRef, keepdim: bool = false, dtype: ScalarKind): RawTensor {.importcpp: "#.mean(@)".}

# median requires std::tuple

func prod*(self: RawTensor): RawTensor {.importcpp: "#.prod()".}
func prod*(self: RawTensor, dtype: ScalarKind): RawTensor {.importcpp: "#.prod(@)".}
func prod*(self: RawTensor, axis: int64, keepdim: bool = false): RawTensor {.importcpp: "#.prod(@)".}
func prod*(self: RawTensor, axis: int64, keepdim: bool = false, dtype: ScalarKind): RawTensor {.importcpp: "#.prod(@)".}

func min*(self: RawTensor): RawTensor {.importcpp: "#.min()".}
func min*(self: RawTensor, axis: int64, keepdim: bool = false): CppTuple2[RawTensor, RawTensor] {.importcpp: "torch::min(@)".}
  ## Returns a tuple (values, indices) of type (TensorT, TensorInt64)
  ## of the minimum values and their index in the specified axis

func max*(self: RawTensor): RawTensor {.importcpp: "#.max()".}
func max*(self: RawTensor, axis: int64, keepdim: bool = false): CppTuple2[RawTensor, RawTensor] {.importcpp: "torch::max(@)".}
  ## Returns a tuple (values, indices) of type (TensorT, TensorInt64)
  ## of the maximum values and their index in the specified axis

func variance*(self: RawTensor, unbiased: bool = true): RawTensor {.importcpp: "#.var(@)".} # can't use `var` because of keyword.
func variance*(self: RawTensor, axis: int64, unbiased: bool = true, keepdim: bool = false): RawTensor {.importcpp: "#.var(@)".}
func variance*(self: RawTensor, axis: IntArrayRef, unbiased: bool = true, keepdim: bool = false): RawTensor {.importcpp: "#.var(@)".}

func stddev*(self: RawTensor, unbiased: bool = true): RawTensor {.importcpp: "#.std(@)".}
func stddev*(self: RawTensor, axis: int64, unbiased: bool = true, keepdim: bool = false): RawTensor {.importcpp: "#.std(@)".}
func stddev*(self: RawTensor, axis: IntArrayRef, unbiased: bool = true, keepdim: bool = false): RawTensor {.importcpp: "#.std(@)".}

# algorithms:
# -----------------------------------------------------------------------
func sort*(self: RawTensor, axis: int64 = -1, descending: bool = false): CppTuple2[RawTensor, RawTensor] {.importcpp: "#.sort(@)".}
  ## Sorts the elements of the input tensor along a given dimension in ascending order by value.
  ## If dim is not given, the last dimension of the input is chosen (dim=-1).
  ## Returns (values, originalIndices) or type (TensorT, TensorInt64)
  ## where originalIndices is the original index of each values (before sorting)
func argsort*(self: RawTensor, axis: int64 = -1, descending: bool = false): RawTensor {.importcpp: "#.argsort(@)".}

func cat*(tensors: ArrayRef[RawTensor], axis: int64 = 0): RawTensor {.importcpp: "torch::cat(@)".}
func flip*(self: RawTensor, dims: IntArrayRef): RawTensor {.importcpp: "#.flip(@)".}

# math
# -----------------------------------------------------------------------
func abs*(self: RawTensor): RawTensor {.importcpp: "#.abs()".}
func absolute*(self: RawTensor): RawTensor {.importcpp: "#.absolute()".}
func angle*(self: RawTensor): RawTensor {.importcpp: "#.angle()".}
func sgn*(self: RawTensor): RawTensor {.importcpp: "#.sgn()".}
func conj*(self: RawTensor): RawTensor {.importcpp: "#.conj()".}
func acos*(self: RawTensor): RawTensor {.importcpp: "#.acos()".}
func arccos*(self: RawTensor): RawTensor {.importcpp: "#.arccos()".}
func acosh*(self: RawTensor): RawTensor {.importcpp: "#.acosh()".}
func arccosh*(self: RawTensor): RawTensor {.importcpp: "#.arccosh()".}
func asinh*(self: RawTensor): RawTensor {.importcpp: "#.asinh()".}
func arcsinh*(self: RawTensor): RawTensor {.importcpp: "#.arcsinh()".}
func atanh*(self: RawTensor): RawTensor {.importcpp: "#.atanh()".}
func arctanh*(self: RawTensor): RawTensor {.importcpp: "#.arctanh()".}
func asin*(self: RawTensor): RawTensor {.importcpp: "#.asin()".}
func arcsin*(self: RawTensor): RawTensor {.importcpp: "#.arcsin()".}
func atan*(self: RawTensor): RawTensor {.importcpp: "#.atan()".}
func arctan*(self: RawTensor): RawTensor {.importcpp: "#.arctan()".}
func cos*(self: RawTensor): RawTensor {.importcpp: "#.cos()".}
func sin*(self: RawTensor): RawTensor {.importcpp: "#.sin()".}
func tan*(self: RawTensor): RawTensor {.importcpp: "#.tan()".}
func exp*(self: RawTensor): RawTensor {.importcpp: "#.exp()".}
func exp2*(self: RawTensor): RawTensor {.importcpp: "#.exp2()".}
func erf*(self: RawTensor): RawTensor {.importcpp: "#.erf()".}
func erfc*(self: RawTensor): RawTensor {.importcpp: "#.erfc()".}
func reciprocal*(self: RawTensor): RawTensor {.importcpp: "#.reciprocal()".}
func neg*(self: RawTensor): RawTensor {.importcpp: "#.neg()".}
func clamp*(self: RawTensor, min, max: Scalar): RawTensor {.importcpp: "#.clamp(@)".}
func clampMin*(self: RawTensor, min: Scalar): RawTensor {.importcpp: "#.clamp_min(@)".}
func clampMax*(self: RawTensor, max: Scalar): RawTensor {.importcpp: "#.clamp_max(@)".}

func dot*(self: RawTensor, other: RawTensor): RawTensor {.importcpp: "#.dot(@)".}

func squeeze*(self: RawTensor): RawTensor {.importcpp: "#.squeeze()".}
func squeeze*(self: RawTensor, axis: int64): RawTensor {.importcpp: "#.squeeze(@)".}
func unsqueeze*(self: RawTensor, axis: int64): RawTensor {.importcpp: "#.unsqueeze(@)".}
func square*(self: RawTensor): RawTensor {.importcpp: "#.square()".}
func sqrt*(self: RawTensor): RawTensor {.importcpp: "#.sqrt()".}
func pow*(self: RawTensor, exponent: RawTensor): RawTensor  {.importcpp: "#.pow(@)".}
func pow*(self: RawTensor, exponent: Scalar): RawTensor {.importcpp: "#.pow(@)".}

# FFT
# -----------------------------------------------------------------------
func fftshift*(self: RawTensor): RawTensor {.importcpp: "torch::fft_fftshift(@)".}
func fftshift*(self: RawTensor, dim: IntArrayRef): RawTensor {.importcpp: "torch::fft_ifftshift(@)".}
func ifftshift*(self: RawTensor): RawTensor {.importcpp: "torch::fft_fftshift(@)".}
func ifftshift*(self: RawTensor, dim: IntArrayRef): RawTensor {.importcpp: "torch::fft_ifftshift(@)".}

let defaultNorm: CppString = initCppString("backward")

func fft*(self: RawTensor, n: int64, dim: int64 = -1, norm: CppString = defaultNorm): RawTensor {.importcpp: "torch::fft_fft(@)".}
## Compute the 1-D Fourier transform
## ``n`` represent Signal length. If given, the input will either be zero-padded or trimmed to this length before computing the FFT.
## ``norm`` can be :
##    * "forward" - normalize by 1/n
##    * "backward" - no normalization
##    * "ortho" - normalize by 1/sqrt(n)
func fft*(self: RawTensor): RawTensor {.importcpp: "torch::fft_fft(@)".}
## Compute the 1-D Fourier transform

func ifft*(self: RawTensor, n: int64, dim: int64 = -1, norm: CppString = defaultNorm): RawTensor {.importcpp: "torch::fft_ifft(@)".}
## Compute the 1-D Fourier transform
## ``n`` represent Signal length. If given, the input will either be zero-padded or trimmed to this length before computing the FFT.
## ``norm`` can be :
##   * "forward" - no normalization
##   * "backward" - normalization by 1/n
##   * "ortho" - normalization by 1/sqrt(n)
func ifft*(self: RawTensor): RawTensor {.importcpp: "torch::fft_ifft(@)".}
## Compute the 1-D Fourier transform

func fft2*(self: RawTensor, s: IntArrayRef, dim: IntArrayRef, norm: CppString = defaultNorm): RawTensor {.importcpp: "torch::fft_fft2(@)".}
## Compute the 2-D Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
## ``norm`` can be :
##    * "forward" - normalize by 1/n
##    * "backward" - no normalization
##    * "ortho" - normalize by 1/sqrt(n)
## With n the logical FFT size: ``n = prod(s)``.
func fft2*(self: RawTensor, s: IntArrayRef): RawTensor {.importcpp: "torch::fft_fft2(@)".}
## Compute the 2-D Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
func fft2*(self: RawTensor): RawTensor {.importcpp: "torch::fft_fft2(@)".}
## Compute the 2-D Fourier transform

func ifft2*(self: RawTensor, s: IntArrayRef, dim: IntArrayRef, norm: CppString = defaultNorm): RawTensor {.importcpp: "torch::fft_ifft2(@)".}
## Compute the 2-D Inverse Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
## ``norm`` can be :
##   * "forward" - no normalization
##   * "backward" - normalization by 1/n
##   * "ortho" - normalization by 1/sqrt(n)
## With n the logical FFT size: ``n = prod(s)``.
func ifft2*(self: RawTensor, s: IntArrayRef): RawTensor {.importcpp: "torch::fft_ifft2(@)".}
## Compute the 2-D Inverse Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
func ifft2*(self: RawTensor): RawTensor {.importcpp: "torch::fft_ifft2(@)".}
## Compute the 2-D Inverse Fourier transform

func fftn*(self: RawTensor, s: IntArrayRef, dim: IntArrayRef, norm: CppString = defaultNorm): RawTensor {.importcpp: "torch::fft_fftn(@)".}
## Compute the N-D Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
## ``norm`` can be :
##    * "forward" normalize by 1/n
##    * "backward" - no normalization
##    * "ortho" normalize by 1/sqrt(n)
## With n the logical FFT size: ``n = prod(s)``.
func fftn*(self: RawTensor, s: IntArrayRef): RawTensor {.importcpp: "torch::fft_fftn(@)".}
## Compute the N-D Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
func fftn*(self: RawTensor): RawTensor {.importcpp: "torch::fft_fftn(@)".}
## Compute the N-D Fourier transform

func ifftn*(self: RawTensor, s: IntArrayRef, dim: IntArrayRef, norm: CppString = defaultNorm): RawTensor {.importcpp: "torch::fft_ifftn(@)".}
## Compute the N-D Inverse Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
## ``norm`` can be :
##   * "forward" - no normalization
##   * "backward" - normalization by 1/n
##   * "ortho" - normalization by 1/sqrt(n)
## With n the logical FFT size: ``n = prod(s)``.
func ifftn*(self: RawTensor, s: IntArrayRef): RawTensor {.importcpp: "torch::fft_ifftn(@)".}
## Compute the N-D Inverse Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
func ifftn*(self: RawTensor): RawTensor {.importcpp: "torch::fft_ifftn(@)".}
## Compute the N-D Inverse Fourier transform

func rfft*(self: RawTensor, n: int64, dim: int64 = -1, norm: CppString = defaultNorm): RawTensor {.importcpp: "torch::fft_rfft(@)".}
## Computes the one dimensional Fourier transform of real-valued input.
func rfft*(self: RawTensor): RawTensor {.importcpp: "torch::fft_rfft(@)".}
## Computes the one dimensional Fourier transform of real-valued input.

func irfft*(self: RawTensor, n: int64, dim: int64 = -1, norm: CppString = defaultNorm): RawTensor {.importcpp: "torch::fft_irfft(@)".}
## Computes the one dimensional Fourier transform of real-valued input.
func irfft*(self: RawTensor): RawTensor {.importcpp: "torch::fft_irfft(@)".}
## Computes the one dimensional Fourier transform of real-valued input.

func rfft2*(self: RawTensor, s: IntArrayRef, dim: IntArrayRef, norm: CppString = defaultNorm): RawTensor {.importcpp: "torch::fft_rfft2(@)".}
## Compute the N-D Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
## ``norm`` can be :
##    * "forward" - normalize by 1/n
##    * "backward" - no normalization
##    * "ortho" - normalize by 1/sqrt(n)
## With n the logical FFT size: ``n = prod(s)``.
func rfft2*(self: RawTensor, s: IntArrayRef): RawTensor {.importcpp: "torch::fft_rfft2(@)".}
## Compute the N-D Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
func rfft2*(self: RawTensor): RawTensor {.importcpp: "torch::fft_rfft2(@)".}
## Compute the N-D Fourier transform

func irfft2*(self: RawTensor, s: IntArrayRef, dim: IntArrayRef, norm: CppString = defaultNorm): RawTensor {.importcpp: "torch::fft_irfft2(@)".}
## Compute the N-D Inverse Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
## ``norm`` can be :
##   * "forward" - no normalization
##   * "backward" - normalization by 1/n
##   * "ortho" - normalization by 1/sqrt(n)
## With n the logical FFT size: ``n = prod(s)``.
func irfft2*(self: RawTensor, s: IntArrayRef): RawTensor {.importcpp: "torch::fft_irfft2(@)".}
## Compute the N-D Inverse Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
func irfft2*(self: RawTensor): RawTensor {.importcpp: "torch::fft_irfft2(@)".}
## Compute the N-D Inverse Fourier transform


func rfftn*(self: RawTensor, s: IntArrayRef, dim: IntArrayRef, norm: CppString = defaultNorm): RawTensor {.importcpp: "torch::fft_rfftn(@)".}
## Compute the N-D Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
## ``norm`` can be :
##    * "forward" - normalize by 1/n
##    * "backward" - no normalization
##    * "ortho" - normalize by 1/sqrt(n)
## With n the logical FFT size: ``n = prod(s)``.
func rfftn*(self: RawTensor, s: IntArrayRef): RawTensor {.importcpp: "torch::fft_rfftn(@)".}
## Compute the N-D Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
func rfftn*(self: RawTensor): RawTensor {.importcpp: "torch::fft_rfftn(@)".}
## Compute the N-D Fourier transform

func irfftn*(self: RawTensor, s: IntArrayRef, dim: IntArrayRef, norm: CppString = defaultNorm): RawTensor {.importcpp: "torch::fft_irfftn(@)".}
## Compute the N-D Inverse Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
## ``norm`` can be :
##   * "forward" - no normalization
##   * "backward" - normalization by 1/n
##   * "ortho" - normalization by 1/sqrt(n)
## With n the logical FFT size: ``n = prod(s)``.
func irfftn*(self: RawTensor, s: IntArrayRef): RawTensor {.importcpp: "torch::fft_irfftn(@)".}
## Compute the N-D Inverse Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
func irfftn*(self: RawTensor): RawTensor {.importcpp: "torch::fft_irfftn(@)".}
## Compute the N-D Inverse Fourier transform

func hfft*(self: RawTensor, n: int64, dim: int64 = -1, norm: CppString = defaultNorm): RawTensor {.importcpp: "torch::hfft(@)".}
## Computes the 1 dimensional FFT of a onesided Hermitian signal.
func hfft*(self: RawTensor): RawTensor {.importcpp: "torch::hfft(@)".}
## Computes the 1 dimensional FFT of a onesided Hermitian signal.
func ihfft*(self: RawTensor, n: int64, dim: int64 = -1, norm: CppString = defaultNorm): RawTensor {.importcpp: "torch::ihfft(@)".}
## Computes the inverse FFT of a real-valued Fourier domain signal.
func ihfft*(self: RawTensor): RawTensor {.importcpp: "torch::ihfft(@)".}
## Computes the inverse FFT of a real-valued Fourier domain signal.

#func convolution*(self: Tensor, weight: Tensor, bias: Tensor, stride, padding, dilation: int64, transposed: bool, outputPadding: int64, groups: int64): Tensor {.importcpp: "torch::convolution(@)".}

{.pop.}
{.pop.}
