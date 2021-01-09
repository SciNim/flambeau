# Flambeau
# Copyright (c) 2020 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  ./tensors,
  ../cpp/std_cpp

# (Almost) raw bindings to PyTorch Data API
# -----------------------------------------------------------------------
#
# This provides almost raw bindings to PyTorch data API.
#
# "Nimification" (camelCase), ergonomic indexing and interoperability with Nim types is left to the "high-level" bindings.
# This should ease searching PyTorch and libtorch documentation,
# and make C++ tutorials easily applicable.

# Headers
# -----------------------------------------------------------------------

{.passC: "-I" & headersPath.}
{.passC: "-I" & torchHeadersPath.}

{.push header: torchHeader.}

# #######################################################################
#
#                         Datasets
#
# #######################################################################
#
# Custom Dataset example: https://github.com/mhubii/libtorch_custom_dataset
# libtorch/include/torch/csrc/api/include/torch/data/datasets/base.h

type
  Example*{.bycopy, importcpp: "torch::data::Example".}
      [Data, Target] = object
    data*: Data
    target*: Target

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

func mnist*(rootPath: cstring, mode = kTrain): Mnist {.constructor, importcpp:"torch::data::datasets::MNIST(@)".}
  ## Loads the MNIST dataset from the `root` path
  ## The supplied `rootpath` should contain the *content* of the unzipped
  ## MNIST dataset, available from http://yann.lecun.com/exdb/mnist.
func get*(dataset: Mnist, index: int): Example[Tensor, Tensor] {.importcpp:"#.get(#)".}
# func size*(dataset: Mnist): optional(int)
func is_train*(): bool {.importcpp:"#.is_train()".}
func images*(dataset: Mnist): lent Tensor {.importcpp: "#.images()".}
  ## Returns all images stacked into a single tensor
func targets*(dataset: Mnist): lent Tensor {.importcpp: "#.targets()".}

# #######################################################################
#
#                         Dataloader
#
# #######################################################################

# #######################################################################
#
#                         Samplers
#
# #######################################################################

# #######################################################################
#
#                         Samplers
#
# #######################################################################
