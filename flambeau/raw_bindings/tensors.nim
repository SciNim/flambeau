# Flambeau
# Copyright (c) 2020 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/[strutils, os],
  ../config

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
# - Nim's `index_fill` and `masked_fill` are mapped to the in-place
#   C++ `index_fill_` and `masked_fill_`.
#   The original out-of-place versions are doing clone+in-place mutation

# #######################################################################
#
#                          C++ Interop
#
# #######################################################################

# Libraries
# -----------------------------------------------------------------------
# I don't think we can do dynamic loading with C++11
# So link directly

const libPath = currentSourcePath.rsplit(DirSep, 1)[0] & "/../libtorch/lib/"

when defined(windows):
  const libSuffix = ".dll"
elif defined(maxosx): # TODO check this
  const libSuffix = ".dylib" # MacOS
else:
  const libSuffix = ".so" # BSD / Linux

# TODO: proper build system on "nimble install" (put libraries in .nimble/bin?)
# if the libPath is not in LD_LIBRARY_PATH
# The libraries won't be loaded at runtime
when true:
  # Not sure what "link" does differently from standard linking,
  # it works, it might even work for both GCC and MSVC
  {.link: libPath & "libc10" & libSuffix.}
  {.link: libPath & "libtorch_cpu" & libSuffix.}

  when UseCuda:
    {.link: libPath & "libtorch_cuda" & libSuffix.}
else:
  # Standard GCC compatible linker
  {.passL: "-L" & libPath & " -lc10 -ltorch_cpu ".}

  when UseCuda:
    {.passL: " -ltorch_cuda ".}

# Headers
# -----------------------------------------------------------------------

const headersPath = currentSourcePath.rsplit(DirSep, 1)[0] & "/../libtorch/include"
const torchHeadersPath = headersPath / "torch/csrc/api/include"
const torchHeader = torchHeadersPath / "torch/torch.h"

{.passC: "-I" & headersPath.}
{.passC: "-I" & torchHeadersPath.}

{.push header: torchHeader.}

# Assumptions
# -----------------------------------------------------------------------
#
# LibTorch is using "ArrayRef" through the codebase in particular
# for shapes and strides.
#
# It has the following definition in
# libtorch/include/c10/util/ArrayRef.h
#
# template <typename T>
# class ArrayRef final {
#  private:
#   /// The start of the array, in an external buffer.
#   const T* Data;
#
#   /// The number of elements.
#   size_type Length;
#
# It is noted that the class does not own the underlying data.
# We can model that in a zero-copy and safely borrow-checked way
# with "openarray[T]"

{.experimental:"views".}

type
  ArrayRef*{.importcpp: "c10::ArrayRef", bycopy.} [T] = object
    # The field are private so we can't use them, but `lent` enforces borrow checking
    p: lent UncheckedArray[T]
    len: csize_t

  IntArrayRef* = ArrayRef[int64]

func data*[T](ar: ArrayRef[T]): lent UncheckedArray[T] {.importcpp: "#.data()".}
func size*(ar: ArrayRef): csize_t {.importcpp: "#.size()".}

func init*[T](AR: type ArrayRef[T], oa: openarray[T]): ArrayRef[T] {.constructor, importcpp: "ArrayRef(@)".}
func init*[T](AR: type ArrayRef[T]): ArrayRef[T] {.constructor, varargs, importcpp: "ArrayRef({@})".}

# #######################################################################
#
#                         Tensor Metadata
#
# #######################################################################

# Backend Device
# -----------------------------------------------------------------------
# libtorch/include/c10/core/DeviceType.h

type
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

# #######################################################################
#
#                            Tensors
#
# #######################################################################

# Tensors
# -----------------------------------------------------------------------

type
  Tensor* {.importcpp: "torch::Tensor", byref.} = object

# Strings & Debugging
# -----------------------------------------------------------------------

proc print*(t: Tensor) {.sideeffect, importcpp: "torch::print(@)".}

# Metadata
# -----------------------------------------------------------------------

func dim*(t: Tensor): int64 {.importcpp: "#.dim()".}
func reset*(t: var Tensor) {.importcpp: "#.reset()".}
func `==`*(a, b: Tensor): bool {.importcpp: "#.is_same(#)".}

func sizes*(a: Tensor): IntArrayRef {.importcpp:"#.sizes()".}
  ## This is Arraymancer and Numpy "shape"
func strides*(a: Tensor): IntArrayRef {.importcpp:"#.strides()".}

func ndimension*(t: Tensor): int64 {.importcpp: "#.ndimension()".}
func nbytes*(t: Tensor): uint {.importcpp: "#.nbytes()".}
func numel*(t: Tensor): int64 {.importcpp: "#.numel()".}
  ## This is Arraymancer and Numpy "size"
func itemsize*(t: Tensor): uint {.importcpp: "#.itemsize()".}
func element_size*(t: Tensor): int64 {.importcpp: "#.element_size()".}

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

# Constructors
# -----------------------------------------------------------------------

# DeviceType and ScalarType are auto-convertible to TensorOptions

func init*(T: type Tensor): Tensor {.constructor,importcpp: "torch::Tensor".}

func from_blob*(data: pointer, sizes: IntArrayRef): Tensor {.importcpp: "torch::from_blob(@)".}
func from_blob*(data: pointer, sizes, strides: IntArrayRef): Tensor {.importcpp: "torch::from_blob(@)".}

func from_blob*(data: pointer, sizes: IntArrayRef, options: TensorOptions): Tensor {.importcpp: "torch::from_blob(@)".}
func from_blob*(data: pointer, sizes: IntArrayRef, scalarKind: ScalarKind): Tensor {.importcpp: "torch::from_blob(@)".}
func from_blob*(data: pointer, sizes: IntArrayRef, device: DeviceKind): Tensor {.importcpp: "torch::from_blob(@)".}

func from_blob*(data: pointer, sizes, strides: IntArrayRef, options: TensorOptions): Tensor {.importcpp: "torch::from_blob(@)".}
func from_blob*(data: pointer, sizes, strides: IntArrayRef, scalarKind: ScalarKind): Tensor {.importcpp: "torch::from_blob(@)".}
func from_blob*(data: pointer, sizes, strides: IntArrayRef, device: DeviceKind): Tensor {.importcpp: "torch::from_blob(@)".}

# Indexing
# -----------------------------------------------------------------------
# libtorch/include/ATen/TensorIndexing.h
# and https://pytorch.org/cppdocs/notes/tensor_indexing.html

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

func index_fill*(a: var Tensor, mask: Tensor, value: Scalar or Tensor) {.importcpp: "#.index_fill_(@)".}
func masked_fill*(a: var Tensor, mask: Tensor, value: Scalar or Tensor) {.importcpp: "#.masked_fill_(@)".}

# Low-level slicing API
# -----------------------------------------------------------------------

type
  TorchSlice* {.importcpp: "torch::indexing::Slice", bycopy.} = object
  # libtorch/include/ATen/TensorIndexing.h

  IndexNone* {.importcpp: "torch::indexing::None", bycopy.} = object
    ## enum class TensorIndexType
  IndexEllipsis* {.importcpp: "torch::indexing::Ellipsis", bycopy.} = object
    ## enum class TensorIndexType

  Ellipsis* = IndexEllipsis
  SomeSlicer* = IndexNone or SomeSignedInt

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
func `+=`*(a: var Tensor, b: Tensor) {.importcpp: "# += #".}
func `+=`*(a: var Tensor, s: Scalar) {.importcpp: "# += #".}
func `-=`*(a: var Tensor, b: Tensor) {.importcpp: "# -= #".}
func `-=`*(a: var Tensor, s: Scalar) {.importcpp: "# -= #".}
func `*=`*(a: var Tensor, b: Tensor) {.importcpp: "# *= #".}
func `*=`*(a: var Tensor, s: Scalar) {.importcpp: "# *= #".}
func `/=`*(a: var Tensor, b: Tensor) {.importcpp: "# /= #".}
func `/=`*(a: var Tensor, s: Scalar) {.importcpp: "# /= #".}
func bitand*(a: var Tensor, s: Tensor) {.importcpp: "# &= #".}
  ## In-place bitwise `and`.
func bitor*(a: var Tensor, s: Tensor) {.importcpp: "# |= #".}
  ## In-place bitwise `or`.
func bitxor*(a: var Tensor, s: Tensor) {.importcpp: "# ^= #".}
  ## In-place bitwise `xor`.

# Functions.h
# -----------------------------------------------------------------------

func eye*(n: int64): Tensor {.importcpp: "torch::eye(@)".}
func eye*(n: int64, options: TensorOptions): Tensor {.importcpp: "torch::eye(@)".}
func eye*(n: int64, scalarKind: ScalarKind): Tensor {.importcpp: "torch::eye(@)".}
func eye*(n: int64, device: DeviceKind): Tensor {.importcpp: "torch::eye(@)".}
