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
  ../libtorch,
  ../cpp/std_cpp,
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

proc manual_seed*(_: type Torch, seed: uint64) {.sideeffect, importcpp:"torch::manual_seed(@)".}
  ## Set torch random number generator seed

# Backends
# -----------------------------------------------------------------------

proc hasCuda*(_: type Torch): bool{.sideeffect, importcpp:"torch::hasCuda()".}
  ## Returns true if libtorch was compiled with CUDA support
proc cuda_is_available*(_: type Torch): bool{.sideeffect, importcpp:"torch::cuda::is_available()".}
  ## Returns true if libtorch was compiled with CUDA support
  ## and at least one CUDA device is available
proc cudnn_is_available*(_: type Torch): bool {.sideeffect, importcpp:"torch::cuda::cudnn_is_available()".}
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
    kUint8 = 0 # kByte
    kInt8 = 1 # kChar
    kInt16 = 2 # kShort
    kInt32 = 3 # kInt
    kInt64 = 4 # kLong
    kFloat16 = 5 # kHalf
    kFloat32 = 6 # kFloat
    kFloat64 = 7 # kDouble
    kComplexF16 = 8 # kComplexHalf
    kComplexF32 = 9 # kComplexFloat
    kComplexF64 = 10 # kComplexDouble
    kBool = 11
    kQint8 = 12 # Quantized int8
    kQuint8 = 13 # Quantized uint8
    kQint32 = 14 # Quantized int32
    kBfloat16 = 15 # Brain float16


  SomeTorchType* = uint8|byte or SomeSignedInt or
                   SomeFloat or Complex[float32] or Complex[float64]

# TensorOptions
# -----------------------------------------------------------------------
# libtorch/include/c10/core/TensorOptions.h

type
  TensorOptions* {.importcpp: "torch::TensorOptions", bycopy.} = object

func init*(T: type TensorOptions): TensorOptions {.constructor,importcpp: "torch::TensorOptions".}

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

proc print*(t: Tensor) {.sideeffect, importcpp: "torch::print(@)".}

# Metadata
# -----------------------------------------------------------------------

func dim*(t: Tensor): int64 {.importcpp: "#.dim()".}
func reset*(t: var Tensor) {.importcpp: "#.reset()".}
func is_same*(a, b: Tensor): bool {.importcpp: "#.is_same(#)".}
  ## Reference equality
  ## Do the tensors use the same memory.

func sizes*(a: Tensor): IntArrayRef {.importcpp:"#.sizes()".}
  ## This is Arraymancer and Numpy "shape"
func strides*(a: Tensor): IntArrayRef {.importcpp:"#.strides()".}

func ndimension*(t: Tensor): int64 {.importcpp: "#.ndimension()".}
func nbytes*(t: Tensor): uint {.importcpp: "#.nbytes()".}
func numel*(t: Tensor): int64 {.importcpp: "#.numel()".}
  ## This is Arraymancer and Numpy "size"
func size*(t: Tensor, axis: int64): int64 {.importcpp: "#.size(#)".}
func itemsize*(t: Tensor): uint {.importcpp: "#.itemsize()".}
func element_size*(t: Tensor): int64 {.importcpp: "#.element_size()".}

# Accessors
# -----------------------------------------------------------------------

func data_ptr*(t: Tensor, T: typedesc[SomeTorchType]): ptr UncheckedArray[T] {.importcpp: "#.data_ptr<'2>(#)".}
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

func has_storage*(t: Tensor): bool {.importcpp: "#.has_storage()".}
func get_device*(t: Tensor): int64 {.importcpp: "#.get_device()".}
func is_cuda*(t: Tensor): bool {.importcpp: "#.is_cuda()".}
func is_hip*(t: Tensor): bool {.importcpp: "#.is_hip()".}
func is_sparse*(t: Tensor): bool {.importcpp: "#.is_sparse()".}
func is_mkldnn*(t: Tensor): bool {.importcpp: "#.is_mkldnn()".}
func is_vulkan*(t: Tensor): bool {.importcpp: "#.is_vulkan()".}
func is_quantized*(t: Tensor): bool {.importcpp: "#.is_quantized()".}
func is_meta*(t: Tensor): bool {.importcpp: "#.is_meta()".}

func cpu*(a: Tensor): Tensor {.importcpp: "#.cpu()".}
func cuda*(a: Tensor): Tensor {.importcpp: "#.cuda()".}
func hip*(a: Tensor): Tensor {.importcpp: "#.hip()".}
func vulkan*(a: Tensor): Tensor {.importcpp: "#.vulkan()".}
func to*(a: Tensor, device: DeviceKind): Tensor {.importcpp: "#.to(#)".}
func to*(a: Tensor, device: Device): Tensor {.importcpp: "#.to(#)".}

# dtype
# -----------------------------------------------------------------------

func to*(a: Tensor, dtype: ScalarKind): Tensor {.importcpp: "#.to(#)".}

# Constructors
# -----------------------------------------------------------------------

# DeviceType and ScalarType are auto-convertible to TensorOptions

func init*(T: type Tensor): Tensor {.constructor,importcpp: "torch::Tensor".}

func from_blob*(data: pointer, sizes: IntArrayRef, options: TensorOptions): Tensor {.importcpp: "torch::from_blob(@)".}
func from_blob*(data: pointer, sizes: IntArrayRef, scalarKind: ScalarKind): Tensor {.importcpp: "torch::from_blob(@)".}
func from_blob*(data: pointer, sizes: IntArrayRef, device: DeviceKind): Tensor {.importcpp: "torch::from_blob(@)".}

func from_blob*(data: pointer, sizes: int64, options: TensorOptions): Tensor {.importcpp: "torch::from_blob(@)".}
func from_blob*(data: pointer, sizes: int64, scalarKind: ScalarKind): Tensor {.importcpp: "torch::from_blob(@)".}
func from_blob*(data: pointer, sizes: int64, device: DeviceKind): Tensor {.importcpp: "torch::from_blob(@)".}

func from_blob*(data: pointer, sizes, strides: IntArrayRef, options: TensorOptions): Tensor {.importcpp: "torch::from_blob(@)".}
func from_blob*(data: pointer, sizes, strides: IntArrayRef, scalarKind: ScalarKind): Tensor {.importcpp: "torch::from_blob(@)".}
func from_blob*(data: pointer, sizes, strides: IntArrayRef, device: DeviceKind): Tensor {.importcpp: "torch::from_blob(@)".}

func empty*(size: IntArrayRef, options: TensorOptions): Tensor {.importcpp:"torch::empty(@)"}
  ## Create an uninitialized tensor of shape `size`
  ## The tensor data must be filled manually
  ##
  ## The output tensor will be row major (C contiguous)
func empty*(size: IntArrayRef, scalarKind: ScalarKind): Tensor {.importcpp:"torch::empty(@)"}
func empty*(size: IntArrayRef, device: DeviceKind): Tensor {.importcpp:"torch::empty(@)"}
  ## Create an uninitialized tensor of shape `size`
  ## The tensor data must be filled manually.
  ##
  ## If device is NOT on CPU make sure to use specialized
  ## copy operations. For example to update on Cuda devices
  ## use cudaMemcpy not a.data[i] = 123
  ##
  ## The output tensor will be row major (C contiguous)

func clone*(a: Tensor): Tensor {.importcpp: "#.clone()".}

# Random sampling
# -----------------------------------------------------------------------

func random_mut*(a: var Tensor, start, stopEx: int64) {.importcpp: "#.random_(@)".}
func randint*(start, stopEx: int64): Tensor {.varargs, importcpp: "torch::randint(#, #, {@})".}
func randint*(start, stopEx: int64, size: IntArrayRef): Tensor {.importcpp: "torch::randint(@)".}

# Indexing
# -----------------------------------------------------------------------
# libtorch/include/ATen/TensorIndexing.h
# and https://pytorch.org/cppdocs/notes/tensor_indexing.html

func item*(a: Tensor, T: typedesc): T {.importcpp: "#.item<'0>()".}
  ## Extract the scalar from a 0-dimensional tensor

# Unsure what those corresponds to in Python
# func `[]`*(a: Tensor, index: Scalar): Tensor {.importcpp: "#[#]".}
# func `[]`*(a: Tensor, index: Tensor): Tensor {.importcpp: "#[#]".}
# func `[]`*(a: Tensor, index: int64): Tensor {.importcpp: "#[#]".}

func index*(a: Tensor): Tensor {.varargs, importcpp: "#.index({@})".}
  ## Tensor indexing. It is recommended
  ## to Nimify this in a high-level wrapper.
  ## `tensor.index(indexers)`

# We can't use the construct `#.index_put_({@}, #)`
# so hardcode sizes,
# 6d seems reasonable, that would be a batch of 3D videos (videoID/batchID, Time, Color Channel, Height, Width, Depth)
# If you need more you likely aren't indexing individual values.

func index_put*(a: var Tensor, i0: auto, val: Scalar or Tensor) {.importcpp: "#.index_put_({#}, #)".}
  ## Tensor mutation at index. It is recommended
  ## to Nimify this in a high-level wrapper.
func index_put*(a: var Tensor, i0, i1: auto, val: Scalar or Tensor) {.importcpp: "#.index_put_({#, #}, #)".}
  ## Tensor mutation at index. It is recommended
  ## to Nimify this in a high-level wrapper.
func index_put*(a: var Tensor, i0, i1, i2: auto, val: Scalar or Tensor) {.importcpp: "#.index_put_({#, #, #}, #)".}
  ## Tensor mutation at index. It is recommended
  ## to Nimify this in a high-level wrapper.
func index_put*(a: var Tensor, i0, i1, i2, i3: auto, val: Scalar or Tensor) {.importcpp: "#.index_put_({#, #, #, #}, #)".}
  ## Tensor mutation at index. It is recommended
  ## to Nimify this in a high-level wrapper.
func index_put*(a: var Tensor, i0, i1, i2, i3, i4: auto, val: Scalar or Tensor) {.importcpp: "#.index_put_({#, #, #, #, #}, #)".}
  ## Tensor mutation at index. It is recommended
  ## to Nimify this in a high-level wrapper.
func index_put*(a: var Tensor, i0, i1, i2, i3, i4, i5: auto, val: Scalar or Tensor) {.importcpp: "#.index_put_({#, #, #, #, #, #}, #)".}
  ## Tensor mutation at index. It is recommended
  ## to Nimify this in a high-level wrapper.

# Fancy Indexing
# -----------------------------------------------------------------------

func index_select*(a: Tensor, axis: int64, indices: Tensor): Tensor {.importcpp: "#.index_select(@)".}
func masked_select*(a: Tensor, mask: Tensor): Tensor {.importcpp: "#.masked_select(@)".}

# PyTorch exposes in-place `index_fill_` and `masked_fill_`
# and out-of-place `index_fill` and `masked_fill`
# that does in-place + clone
# we only exposes the in-place version.

func index_fill_mut*(a: var Tensor, mask: Tensor, value: Scalar or Tensor) {.importcpp: "#.index_fill_(@)".}
func masked_fill_mut*(a: var Tensor, mask: Tensor, value: Scalar or Tensor) {.importcpp: "#.masked_fill_(@)".}

# Shapeshifting
# -----------------------------------------------------------------------

func reshape*(a: Tensor): Tensor {.varargs, importcpp: "#.reshape({@})".}
func view*(a: Tensor): Tensor {.varargs, importcpp: "#.reshape({@})".}

# Automatic Differentiation
# -----------------------------------------------------------------------

func backward*(a: var Tensor){.importcpp: "#.backward()".}

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

func `not`*(t: Tensor): Tensor {.importcpp: "~#".}
func `-`*(t: Tensor): Tensor {.importcpp: "-#".}

func `+`*(a: Tensor, b: Tensor): Tensor {.importcpp: "# + #".}
func `-`*(a: Tensor, b: Tensor): Tensor {.importcpp: "# - #".}
func `*`*(a: Tensor, b: Tensor): Tensor {.importcpp: "# * #".}

func `+=`*(a: var Tensor, b: Tensor) {.importcpp: "# += #".}
func `+=`*(a: var Tensor, s: Scalar) {.importcpp: "# += #".}
func `-=`*(a: var Tensor, b: Tensor) {.importcpp: "# -= #".}
func `-=`*(a: var Tensor, s: Scalar) {.importcpp: "# -= #".}
func `*=`*(a: var Tensor, b: Tensor) {.importcpp: "# *= #".}
func `*=`*(a: var Tensor, s: Scalar) {.importcpp: "# *= #".}
func `/=`*(a: var Tensor, b: Tensor) {.importcpp: "# /= #".}
func `/=`*(a: var Tensor, s: Scalar) {.importcpp: "# /= #".}

func `and`*(a: Tensor, b: Tensor): Tensor {.importcpp: "#.bitwise_and(#)".}
  ## bitwise `and`.
func `or`*(a: Tensor, b: Tensor): Tensor {.importcpp: "#.bitwise_or(#)".}
  ## bitwise `or`.
func `xor`*(a: Tensor, b: Tensor): Tensor {.importcpp: "#.bitwise_xor(#)".}
  ## bitwise `xor`.

func bitand_mut*(a: var Tensor, s: Tensor) {.importcpp: "#.bitwise_and_(#)".}
  ## In-place bitwise `and`.
func bitor_mut*(a: var Tensor, s: Tensor) {.importcpp: "#.bitwise_or_(#)".}
  ## In-place bitwise `or`.
func bitxor_mut*(a: var Tensor, s: Tensor) {.importcpp: "#.bitwise_xor_(#)".}
  ## In-place bitwise `xor`.

func eq*(a, b: Tensor): Tensor {.importcpp: "#.eq(#)".}
  ## Equality of each tensor values
func equal*(a, b: Tensor): bool {.importcpp: "#.equal(#)".}
template `==`*(a, b: Tensor): bool =
  a.equal(b)
# Functions.h
# -----------------------------------------------------------------------

func toType*(t: Tensor, dtype: ScalarKind): Tensor {.importcpp: "#.toType(@)".}
func toSparse*(t: Tensor): Tensor {.importcpp: "#.to_sparse()".}
func toSparse*(t: Tensor, sparseDim: int64): Tensor {.importcpp: "#.to_sparse(@)".}

func eye*(n: int64): Tensor {.importcpp: "torch::eye(@)".}
func eye*(n: int64, options: TensorOptions): Tensor {.importcpp: "torch::eye(@)".}
func eye*(n: int64, scalarKind: ScalarKind): Tensor {.importcpp: "torch::eye(@)".}
func eye*(n: int64, device: DeviceKind): Tensor {.importcpp: "torch::eye(@)".}

func zeros*(dim: int64): Tensor {.importcpp: "torch::zeros(@)".}
func zeros*(dim: IntArrayRef): Tensor {.importcpp: "torch::zeros(@)".}
func zeros*(dim: IntArrayRef, options: TensorOptions): Tensor {.importcpp: "torch::zeros(@)".}
func zeros*(dim: IntArrayRef, scalarKind: ScalarKind): Tensor {.importcpp: "torch::zeros(@)".}
func zeros*(dim: IntArrayRef, device: DeviceKind): Tensor {.importcpp: "torch::zeros(@)".}

func add*(t: Tensor, other: Tensor, alpha: Scalar = 1): Tensor {.importcpp: "#.add(@)".}
func add*(t: Tensor, other: Scalar, alpha: Scalar = 1): Tensor {.importcpp: "#.add(@)".}
func addmv*(t: Tensor, mat: Tensor, vec: Tensor, beta: Scalar = 1, alpha: Scalar = 1): Tensor {.importcpp: "#.addmv(@)".}
func addmm*(t, mat1, mat2: Tensor, beta: Scalar = 1, alpha: Scalar = 1): Tensor {.importcpp: "#.addmm(@)".}
func mm*(t, other: Tensor): Tensor {.importcpp: "#.mm(@)".}
func matmul*(t, other: Tensor): Tensor {.importcpp: "#.matmul(@)".}
func bmm*(t, other: Tensor): Tensor {.importcpp: "#.bmm(@)".}

func luSolve*(t, data, pivots: Tensor): Tensor {.importcpp: "#.lu_solve(@)".}

func qr*(t: Tensor, some: bool = true): CppTuple2[Tensor, Tensor] {.importcpp: "#.qr(@)".}
  ## Returns a tuple:
  ## - Q of shape (∗,m,k)
  ## - R of shape (∗,k,n)
  ## with k=min(m,n) if some is true otherwise k=m
  ##
  ## The QR decomposition is batched over dimension(s) *
  ## t = QR

# addr?
func all*(t: Tensor, axis: int64): Tensor {.importcpp: "#.all(@)".}
func all*(t: Tensor, axis: int64, keepdim: bool): Tensor {.importcpp: "#.all(@)".}
func allClose*(t, other: Tensor, rtol: float64 = 1e-5, abstol: float64 = 1e-8, equalNan: bool = false): bool {.importcpp: "#.allclose(@)".}
func any*(t: Tensor, axis: int64): Tensor {.importcpp: "#.any(@)".}
func any*(t: Tensor, axis: int64, keepdim: bool): Tensor {.importcpp: "#.any(@)".}
func argmax*(t: Tensor): Tensor {.importcpp: "#.argmax()".}
func argmax*(t: Tensor, axis: int64, keepdim: bool = false): Tensor {.importcpp: "#.argmax(@)".}
func argmin*(t: Tensor): Tensor {.importcpp: "#.argmin()".}
func argmin*(t: Tensor, axis: int64, keepdim: bool = false): Tensor {.importcpp: "#.argmin(@)".}

# aggregate

# sum needs wrapper procs/templates to allow for using nim arrays and single axis.
func sum*(t: Tensor): Tensor {.importcpp: "#.sum()".}
func sum*(t: Tensor, dtype: ScalarKind): Tensor {.importcpp: "#.sum(@)".}
func sum*(t: Tensor, axis: int64, keepdim: bool = false): Tensor {.importcpp: "#.sum(@)".}
func sum*(t: Tensor, axis: int64, keepdim: bool = false, dtype: ScalarKind): Tensor {.importcpp: "#.sum(@)".}
func sum*(t: Tensor, axis: IntArrayRef, keepdim: bool = false): Tensor {.importcpp: "#.sum(@)".}
func sum*(t: Tensor, axis: IntArrayRef, keepdim: bool = false, dtype: ScalarKind): Tensor {.importcpp: "#.sum(@)".}

# mean as well
func mean*(t: Tensor): Tensor {.importcpp: "#.mean()".}
func mean*(t: Tensor, dtype: ScalarKind): Tensor {.importcpp: "#.mean(@)".}
func mean*(t: Tensor, axis: int64, keepdim: bool = false): Tensor {.importcpp: "#.mean(@)".}
func mean*(t: Tensor, axis: int64, keepdim: bool = false, dtype: ScalarKind): Tensor {.importcpp: "#.mean(@)".}
func mean*(t: Tensor, axis: IntArrayRef, keepdim: bool = false): Tensor {.importcpp: "#.mean(@)".}
func mean*(t: Tensor, axis: IntArrayRef, keepdim: bool = false, dtype: ScalarKind): Tensor {.importcpp: "#.mean(@)".}

# median requires std::tuple

func prod*(t: Tensor): Tensor {.importcpp: "#.prod()".}
func prod*(t: Tensor, dtype: ScalarKind): Tensor {.importcpp: "#.prod(@)".}
func prod*(t: Tensor, axis: int64, keepdim: bool = false): Tensor {.importcpp: "#.prod(@)".}
func prod*(t: Tensor, axis: int64, keepdim: bool = false, dtype: ScalarKind): Tensor {.importcpp: "#.prod(@)".}

func min*(t: Tensor): Tensor {.importcpp: "#.min()".}
func min*(t: Tensor, axis: int64, keepdim: bool = false): CppTuple2[Tensor, Tensor] {.importcpp: "torch::min(@)".}
  ## Returns a tuple (values, indices) of type (TensorT, TensorInt64)
  ## of the minimum values and their index in the specified axis

func max*(t: Tensor): Tensor {.importcpp: "#.max()".}
func max*(t: Tensor, axis: int64, keepdim: bool = false): CppTuple2[Tensor, Tensor] {.importcpp: "torch::max(@)".}
  ## Returns a tuple (values, indices) of type (TensorT, TensorInt64)
  ## of the maximum values and their index in the specified axis

func variance*(t: Tensor, unbiased: bool = true): Tensor {.importcpp: "#.var(@)".} # can't use `var` because of keyword.
func variance*(t: Tensor, axis: int64, unbiased: bool = true, keepdim: bool = false): Tensor {.importcpp: "#.var(@)".}
func variance*(t: Tensor, axis: IntArrayRef, unbiased: bool = true, keepdim: bool = false): Tensor {.importcpp: "#.var(@)".}

func stddev*(t: Tensor, unbiased: bool = true): Tensor {.importcpp: "#.std(@)".}
func stddev*(t: Tensor, axis: int64, unbiased: bool = true, keepdim: bool = false): Tensor {.importcpp: "#.std(@)".}
func stddev*(t: Tensor, axis: IntArrayRef, unbiased: bool = true, keepdim: bool = false): Tensor {.importcpp: "#.std(@)".}

# algorithms:

func sort*(t: Tensor, axis: int64 = -1, descending: bool = false): CppTuple2[Tensor, Tensor] {.importcpp: "#.sort(@)".}
  ## Sorts the elements of the input tensor along a given dimension in ascending order by value.
  ## If dim is not given, the last dimension of the input is chosen (dim=-1).
  ## Returns (values, originalIndices) or type (TensorT, TensorInt64)
  ## where originalIndices is the original index of each values (before sorting)
func argsort*(t: Tensor, axis: int64 = -1, descending: bool = false): Tensor {.importcpp: "#.argsort(@)".}

# math
func abs*(t: Tensor): Tensor {.importcpp: "#.abs()".}
func absolute*(t: Tensor): Tensor {.importcpp: "#.absolute()".}
func angle*(t: Tensor): Tensor {.importcpp: "#.angle()".}
func sgn*(t: Tensor): Tensor {.importcpp: "#.sgn()".}
func conj*(t: Tensor): Tensor {.importcpp: "#.conj()".}
func acos*(t: Tensor): Tensor {.importcpp: "#.acos()".}
func arccos*(t: Tensor): Tensor {.importcpp: "#.arccos()".}
func acosh*(t: Tensor): Tensor {.importcpp: "#.acosh()".}
func arccosh*(t: Tensor): Tensor {.importcpp: "#.arccosh()".}
func asinh*(t: Tensor): Tensor {.importcpp: "#.asinh()".}
func arcsinh*(t: Tensor): Tensor {.importcpp: "#.arcsinh()".}
func atanh*(t: Tensor): Tensor {.importcpp: "#.atanh()".}
func arctanh*(t: Tensor): Tensor {.importcpp: "#.arctanh()".}
func asin*(t: Tensor): Tensor {.importcpp: "#.asin()".}
func arcsin*(t: Tensor): Tensor {.importcpp: "#.arcsin()".}
func atan*(t: Tensor): Tensor {.importcpp: "#.atan()".}
func arctan*(t: Tensor): Tensor {.importcpp: "#.arctan()".}
func cos*(t: Tensor): Tensor {.importcpp: "#.cos()".}
func sin*(t: Tensor): Tensor {.importcpp: "#.sin()".}
func tan*(t: Tensor): Tensor {.importcpp: "#.tan()".}
func exp*(t: Tensor): Tensor {.importcpp: "#.exp()".}
func exp2*(t: Tensor): Tensor {.importcpp: "#.exp2()".}
func erf*(t: Tensor): Tensor {.importcpp: "#.erf()".}
func erfc*(t: Tensor): Tensor {.importcpp: "#.erfc()".}
func reciprocal*(t: Tensor): Tensor {.importcpp: "#.reciprocal()"}
func neg*(t: Tensor): Tensor {.importcpp: "#.neg()".}
func clamp*(t: Tensor, min, max: Scalar): Tensor {.importcpp: "#.clamp(@)".}
func clampMin*(t: Tensor, min: Scalar): Tensor {.importcpp: "#.clamp_min(@)".}
func clampMax*(t: Tensor, max: Scalar): Tensor {.importcpp: "#.clamp_max(@)".}

func dot*(t: Tensor, other: Tensor): Tensor {.importcpp: "#.dot(@)".}

func squeeze*(t: Tensor): Tensor {.importcpp: "#.squeeze()".}
func squeeze*(t: Tensor, axis: int64): Tensor {.importcpp: "#.squeeze(@)".}
func unsqueeze*(t: Tensor, axis: int64): Tensor {.importcpp: "#.unsqueeze(@)".}

func fft*(t: Tensor): Tensor {.importcpp: "torch::fft_fft(@)".}
func fft*(t: Tensor, n: int64, axis: int64 = -1): Tensor {.importcpp: "torch::fft_fft(@)".}
func fft*(t: Tensor, n: int64, axis: int64 = -1, norm: CppString): Tensor {.importcpp: "torch::fft_fft(@)".}

#func convolution*(t: Tensor, weight: Tensor, bias: Tensor, stride, padding, dilation: int64, transposed: bool, outputPadding: int64, groups: int64): Tensor {.importcpp: "torch::convolution(@)".}
