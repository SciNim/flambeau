# Flambeau
# Copyright (c) 2020 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  ../cpp/std_cpp,
  ../libtorch,
  ./c10, ./tensors

# (Almost) raw bindings to PyTorch Data API
# -----------------------------------------------------------------------
#
# This provides almost raw bindings to PyTorch data API.
#
# "Nimification" (camelCase), ergonomic indexing and interoperability with Nim types is left to the "high-level" bindings.
# This should ease searching PyTorch and libtorch documentation,
# and make C++ tutorials easily applicable.

# C++ interop
# -----------------------------------------------------------------------

{.push cdecl.}
{.push header: torchHeader.}

# #######################################################################
#
#                         Iterators
#
# #######################################################################
#
# /home/beta/Programming/Nim/flambeau/vendor/libtorch/include/torch/csrc/api/include/torch/data/iterator.h

type
  TorchDataIterator*[Batch]
    {.bycopy, importcpp: "torch::data::Iterator".}
    = object

func next*(it: var TorchDataIterator) {.importcpp: "(++#)".}
func get*[Batch](it: var TorchDataIterator[Batch]): Batch {.importcpp: "(*#)".}
  # TODO: this should be lent?
func `==`*(it1, it2: TorchDataIterator): bool {.importcpp: "# == #".}

# #######################################################################
#
#                         Samplers
#
# #######################################################################

type
  # TODO: https://github.com/nim-lang/Nim/issues/16653
  #   generics + {.inheritable.} doesn't work
  Sampler* # [BatchRequest]
      {.bycopy, pure, inheritable,
      importcpp: "torch::data::samplers::Sampler".}
    = object

  RandomSampler*
      {.bycopy, pure,
      importcpp: "torch::data::samplers::RandomSampler".}
    = object of Sampler

  SequentialSampler*
      {.bycopy, pure,
      importcpp: "torch::data::samplers::SequentialSampler".}
    = object of Sampler

func init*(T: type SequentialSampler): T {.constructor, importcpp: "torch::data::samplers::SequentialSampler()".}

# #######################################################################
#
#                         Example
#
# #######################################################################

type
  Example*[Data, Target]
    {.bycopy, importcpp: "torch::data::Example".}
    = object
    ## An example maps data to a target label
    ## for classification
    data*: Data
    target*: Target

# #######################################################################
#
#                         Transforms
#
# #######################################################################

type
  # libtorch/include/torch/csrc/api/include/torch/data/transforms/base.h
  # --------------------------------------------------------------------

  # TODO: https://github.com/nim-lang/Nim/issues/16653
  #   generics + {.inheritable.} doesn't work
  BatchTransform* # [BatchRequest]
      {.bycopy, pure, inheritable,
      importcpp: "torch::data::transforms::BatchTransform".}
    = object

  Transform*[Input, Output]
      {.bycopy, pure,
      importcpp: "torch::data::transforms::Transform".}
    = object of BatchTransform # [CppVector[Input], CppVector[Output]]

  # libtorch/include/torch/csrc/api/include/torch/data/transforms/lambda.h
  # ----------------------------------------------------------------------

  BatchLambda*[Input, Output]
      {.bycopy, pure,
      importcpp: "torch::data::transforms::BatchLambda".}
    = object of BatchTransform
    ## A `BatchTransform` that applies a user-provided functor to a batch.

  Lambda*[Input, Output]
      {.bycopy, pure,
      importcpp: "torch::data::transforms::Lambda".}
    = object
    ## A `Transform` that applies a user-provided functor to individual examples.

  # libtorch/include/torch/csrc/api/include/torch/data/transforms/collate.h
  # -----------------------------------------------------------------------

  # TODO: https://github.com/nim-lang/Nim/issues/16653
  #   generics + {.inheritable.} doesn't work
  Collation*[BatchType, T]
    {.importcpp:"torch::data::transforms::Collation".}
    = BatchTransform # [BatchType, T]

  Collate*[BatchType, T]
    {.importcpp:"torch::data::transforms::Collate".}
    = BatchLambda[BatchType, T]

  # libtorch/include/torch/csrc/api/include/torch/data/transforms/stack.h
  # ---------------------------------------------------------------------
  Stack*[E]
    {.importcpp:"torch::data::transforms::Stack".}
    = object of Collate[CppVector[E], E]
    ## A `Collation` for `Example<Tensor, Tensor>` types that stacks all data
    ## tensors into one tensor, and all target (label) tensors into one tensor.
    ##
    ## or
    ##
    ## A `Collation` for `Example<Tensor, NoTarget>` types that stacks all data
    ## tensors into one tensor.

func init*(S: type Stack): S {.constructor, importcpp: "torch::data::transforms::Stack<>()".}

type
  # libtorch/include/torch/csrc/api/include/torch/data/transforms/tensor.h
  # ----------------------------------------------------------------------
  TensorTransform*[Target]
      {.bycopy, pure,
      importcpp: "torch::data::transforms::TensorTransform".}
    = object of Transform[Example[Tensor, Target], Example[Tensor, Target]]

  TensorLambda*[Target]
      {.bycopy, pure,
      importcpp: "torch::data::transforms::TensorLambda".}
    = object of TensorTransform[Target]

  Normalize*[Target]
      {.bycopy, pure,
      importcpp: "torch::data::transforms::Normalize".}
    = object of TensorTransform[Target]
    ## Normalized input tensors
    ## by substracting the supplied mean
    ## and dividing by the given standard deviation

func init*(T: type Normalize, mean, stddev: float64): T {.constructor, importcpp: "torch::data::transforms::Normalize<>(@)".}
func init*(T: type Normalize, mean, stddev: ArrayRef[float64]): T {.constructor, importcpp: "torch::data::transforms::Normalize<>(@)".}

# #######################################################################
#
#                         Datasets
#
# #######################################################################
#
# Custom Dataset example: https://github.com/mhubii/libtorch_custom_dataset
# libtorch/include/torch/csrc/api/include/torch/data/datasets/base.h

type
  # TODO: https://github.com/nim-lang/Nim/issues/16653
  #   generics + {.inheritable.} doesn't work
  BatchDataset*
        {.bycopy, pure, inheritable,
        importcpp: "torch::data::datasets::BatchDataset".}
        # [Self, Batch, BatchRequest] # TODO: generic inheritable https://github.com/nim-lang/Nim/issues/16653
      = object
    ## A BatchDataset type
    ## Self: is the class type that implements the Dataset API
    ##   (using the Curious Recurring Template Pattern in underlying C++)
    ## Batch is by default the type CppVector[T]
    ##   with T being Example[Data, Target]
    ## BatchRequest is by default ArrayRef[csize_t]

  # TODO: https://github.com/nim-lang/Nim/issues/16653
  #   generics + {.inheritable.} doesn't work
  # TODO: https://github.com/nim-lang/Nim/issues/16655
  # CRTP + importcpp don't work
  Dataset*
    {.bycopy, pure,
    importcpp: "torch::data::datasets::Dataset".}
    # [Self, Batch]
      = object of BatchDataset # [Self, Batch, ArrayRef[csize_t]]
    ## A Dataset type
    ## Self: is the class type that implements the Dataset API
    ##   (using the Curious Recurring Template Pattern in underlying C++)
    ## Batch is by default the type CppVector[T]
    ##   with T being Example[Data, Target]

  # TODO: https://github.com/nim-lang/Nim/issues/16655
  # CRTP + importcpp don't work
  Mnist*
    {.bycopy, pure,
    importcpp: "torch::data::datasets::MNIST".}
    = object of Dataset # [Mnist, CppVector[Example[Tensor, Tensor]]]
    ## The MNIST dataset
    ## http://yann.lecun.com/exdb/mnist

  MnistMode* {.size:sizeof(cint),
      importcpp:"torch::data::datasets::MNIST::Mode".} = enum
    ## Select the train or test mode of the Mnist data
    kTrain = 0
    kTest = 1

func is_stateful*(D: type BatchDataset): bool {.importcpp: "'1::is_stateful".}
func size*(dataset: BatchDataset): Optional[int] {.importcpp:"#.size()".}
  ## Returns the size in an optional.
  ## If the dataset has no size (stream?)
  ## the optional is "None"

func mnist*(rootPath: cstring, mode = kTrain): Mnist {.constructor, importcpp:"torch::data::datasets::MNIST(@)".}
  ## Loads the MNIST dataset from the `root` path
  ## The supplied `rootpath` should contain the *content* of the unzipped
  ## MNIST dataset, available from http://yann.lecun.com/exdb/mnist.
func get*(dataset: Mnist, index: int): Example[Tensor, Tensor] {.importcpp:"#.get(#)".}
func is_train*(): bool {.importcpp:"#.is_train()".}
func images*(dataset: Mnist): lent Tensor {.importcpp: "#.images()".}
  ## Returns all images stacked into a single tensor
func targets*(dataset: Mnist): lent Tensor {.importcpp: "#.targets()".}

# libtorch/include/torch/csrc/api/include/torch/data/datasets/map.h
# libtorch/include/torch/csrc/api/include/torch/data/datasets/base.h
# ----------------------------------------------------------------------
# The inner details are in map.h but the public view is in base.h
type
  MapDataset* [SourceDataset, AppliedTransform]
    {.bycopy, pure, importcpp: "torch::data::datasets::MapDataset".}
    = object of BatchDataset

func map*[DatasetType; TransformType: not type](
       dataset: sink DatasetType,
       transform: sink TransformType
  ): MapDataset[DatasetType, TransformType] {.importcpp: "#.map(#)".}

func map*[DatasetType](
       dataset: sink DatasetType,
       transformType: typedesc
  ): MapDataset[DatasetType, transformType] {.inline, nimcall.}=
  # TODO: bad C++ codegen, the typedesc arg disappears
  map(dataset, init(transformType))

# #######################################################################
#
#                         Dataloader
#
# #######################################################################

when not compileOption("threads"):
  {.error: "The data API is multithreaded, use the --threads:on compilation flag".}

type
  # TODO: https://github.com/nim-lang/Nim/issues/16653
  #   generics + {.inheritable.} doesn't work
  DataLoaderBase*
        {.byref, pure, inheritable,
        importcpp: "torch::data::datasets::BatchDataset".}
        # [Dataset, Batch, BatchRequest] # TODO: generic inheritable https://github.com/nim-lang/Nim/issues/16653
      = object

  StatelessDataLoader*[D, S] # Dataset, Sampler
        {.byref, pure,
        importcpp: "torch::data::StatelessDataLoader".}
      = object of DataLoaderBase

  StatefulDataLoader*[D] # Dataset
        {.byref, pure,
        importcpp: "torch::data::StatefulDataLoader".}
      = object of DataLoaderBase

  DataLoaderOptions*
        {.byref,
        importcpp: "torch::data::DataLoaderOptions".} = object
    # TODO: multithreaded batch support

# TODO: because of https://github.com/nim-lang/Nim/issues/16653
#       and https://github.com/nim-lang/Nim/issues/16655
#       BatchDataset and Dataset have no generics attached
#       and so we can't infer their Iterator type :/
func start*(dl: StatelessDataLoader
       ): TorchDataIterator[Example[Tensor, Tensor]]
  {.importcpp: "#.begin()".}
  ## Start an iterator
  ## Note: due to compiler bugs with C++ interop
  ##       we can't attach the DataLoaderBase generic type,
  ##       and so the output is fixed to Example[Tensor, Tensor]
  ##       which is the output of the Stack transform

func stop*(dl: StatelessDataLoader
       ): TorchDataIterator[Example[Tensor, Tensor]]
       {.importcpp: "#.end()".}
  ## Returns a sentinel value that denotes
  ## the end of an iterator

func start*[D, S](
         dl: CppUniquePtr[StatelessDataLoader[D, S]]
       ): TorchDataIterator[Example[Tensor, Tensor]]
  {.importcpp: "#->begin()".}
  ## Start an iterator
  ## Note: due to compiler bugs with C++ interop
  ##       we can't attach the DataLoaderBase generic type,
  ##       and so the output is fixed to Example[Tensor, Tensor]
  ##       which is the output of the Stack transform
  ##
  ## Overload as StatelessDataLoader has no default constructors
  ## So we don't want Nim to use temporaries

func stop*[D, S](
         dl: CppUniquePtr[StatelessDataLoader[D, S]]
       ): TorchDataIterator[Example[Tensor, Tensor]]
       {.importcpp: "#->end()".}
  ## Returns a sentinel value that denotes
  ## the end of an iterator
  ##
  ## Overload as StatelessDataLoader has no default constructors
  ## So we don't want Nim to use temporaries

iterator items*(dl: StatelessDataLoader or CppUniquePtr[StatelessDataLoader]): Example[Tensor, Tensor] =
  # TODO: lent Example[Tensor, Tensor],
  #   borrow checker complains about 'cur' escaping it's frame
  #   but `cur.get()` already returns a borrowed view
  var cur = dl.start()
  let stop = dl.stop()
  while cur != stop:
    yield cur.get()
    cur.next()

iterator pairs*(dl: StatelessDataLoader or CppUniquePtr[StatelessDataLoader]): tuple[index: int, value: Example[Tensor, Tensor]] =
  # TODO: lent Example[Tensor, Tensor]
  #   borrow checker complains about 'cur' escaping it's frame
  #   but `cur.get()` already returns a borrowed view
  var cur = dl.start()
  let stop = dl.stop()
  var index = 0
  while cur != stop:
    yield (index, cur.get())
    inc index
    cur.next()

# Note make_data_loader is using `enable_if`
# to dispatch between either
# a StatelessDataLoader or a StatefulDataLoader
# A StatefulDataLoader uses optional<T> instead of plain T
#
# libtorch/include/torch/csrc/api/include/torch/data/datasets/base.h
# -> line 30, 32 and 45
#
# This can be represented in Nim with
# type DataLoader[Dataset, Sampler] = object
#   when Dataset.Batch.Element is TorchOptional:
#     dl: CppUniquePtr[StatefulDataLoader[Dataset, Sampler]] # Sampler = DummySampler for Stateful
#   else:
#     dl: CppUniquePtr[StatelessDataLoader[Dataset, Sampler]]
#
# Or
#
# func make_data_loader*[D: BatchDataset and not Optional](...): CppUniquePtr[StatelessDataLoader[D, RandomSampler]]
# func make_data_loader*[D: BatchDataset and Optional](...): CppUniquePtr[StatefulDataLoader[D]]
#
# potentially using concept for constraints if higher-kinded constraints don't work
#
# For now we assume Stateless datasets

func make_data_loader*[D: BatchDataset](
       dataset: D
  ): CppUniquePtr[StatelessDataLoader[D, RandomSampler]] {.
  importcpp: "torch::data::make_data_loader(#)".}

func make_data_loader*[D: BatchDataset](
       dataset: D,
       options: DataLoaderOptions
  ): CppUniquePtr[StatelessDataLoader[D, RandomSampler]] {.
  importcpp: "torch::data::make_data_loader(@)".}

func make_data_loader*[D: BatchDataset](
       dataset: D,
       batch_size: csize_t
  ): CppUniquePtr[StatelessDataLoader[D, RandomSampler]] {.
  importcpp: "torch::data::make_data_loader(@)".}

func make_data_loader*[D: BatchDataset; S: Sampler](
       dataset: D,
       sampler: S,
  ): CppUniquePtr[StatelessDataLoader[D, S]] {.
  importcpp: "torch::data::make_data_loader(@)".}

func make_data_loader*[D: BatchDataset; S: Sampler](
       dataset: D,
       sampler: S,
       options: DataLoaderOptions
  ): CppUniquePtr[StatelessDataLoader[D, S]] {.
  importcpp: "torch::data::make_data_loader(@)".}

func make_data_loader*[D: BatchDataset; S: Sampler](
       dataset: D,
       sampler: S,
       batch_size: csize_t
  ): CppUniquePtr[StatelessDataLoader[D, S]] {.
  importcpp: "torch::data::make_data_loader(@)".}

func make_data_loader*[D: BatchDataset; S: Sampler](
       SamplerType: type S,
       dataset: D
  ): CppUniquePtr[StatelessDataLoader[D, SamplerType]] {.
  importcpp: "torch::data::make_data_loader<'*1>(#)".}

func make_data_loader*[D: BatchDataset; S: Sampler](
       SamplerType: type S,
       dataset: D,
       options: DataLoaderOptions
  ): CppUniquePtr[StatelessDataLoader[D, SamplerType]] {.
  importcpp: "torch::data::make_data_loader<'*1>(@)".}

func make_data_loader*[D: BatchDataset; S: Sampler](
       SamplerType: type S,
       dataset: D,
       batch_size: csize_t
  ): CppUniquePtr[StatelessDataLoader[D, SamplerType]] {.
  importcpp: "torch::data::make_data_loader<'*1>(@)".}
