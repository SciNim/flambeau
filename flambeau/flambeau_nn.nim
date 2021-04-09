# Flambeau
# Copyright (c) 2020 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Raw exports
# ----------------------------------------------------------------

import ./flambeau/raw/bindings/rawtensors
  except # TODO, don't export associated proc either
    # ArrayRef,
    TensorOptions,
    TorchSlice, IndexNone, IndexEllipsis, SomeSlicer, torchSlice

export rawtensors # TODO, don't export low-level procs and types like C++ slices.

import ./flambeau/raw/bindings/neural_nets
  except LinearOptions
export neural_nets

import ./flambeau/raw/bindings/optimizers
  except OptimizerOptions
export optimizers

import ./flambeau/raw/bindings/data_api
  except
    TorchDataIterator,
    DataLoaderBase, DataLoaderOptions
export data_api

import ./flambeau/raw/bindings/serialize
export serialize

# C++ Standard Library
# ----------------------------------------------------------------

import ./flambeau/raw/cpp/[std_cpp, emitters]
export std_cpp, emitters

# Convenience helpers
# ----------------------------------------------------------------

import ./flambeau/glucose/[indexing, interop, moduleSugar]
export indexing, interop, moduleSugar

import ./flambeau/raw/bindings/c10
export c10

