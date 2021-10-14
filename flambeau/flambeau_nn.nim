# Flambeau
# Copyright (c) 2020 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Raw exports
# ----------------------------------------------------------------

import ./raw/bindings/rawtensors
  except # TODO, don't export associated proc either
    # ArrayRef,
    TensorOptions,
    TorchSlice, IndexNone, IndexEllipsis, SomeSlicer, torchSlice

export rawtensors # TODO, don't export low-level procs and types like C++ slices.

import ./raw/bindings/neural_nets
  except LinearOptions
export neural_nets

import ./raw/bindings/optimizers
  except OptimizerOptions
export optimizers

import ./raw/bindings/data_api
  except
    TorchDataIterator,
    DataLoaderBase, DataLoaderOptions
export data_api

import ./raw/bindings/serialize
export serialize

# C++ Standard Library
# ----------------------------------------------------------------

import ./raw/cpp/[std_cpp, emitters]
export std_cpp, emitters

import cppstl/[std_smartptrs]
export std_smartptrs

# Convenience helpers
# ----------------------------------------------------------------

import ./raw/sugar/[indexing, interop, moduleSugar]
export indexing, interop, moduleSugar

import ./raw/bindings/c10
export c10
