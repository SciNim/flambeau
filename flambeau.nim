# Flambeau
# Copyright (c) 2020 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Raw exports
# ----------------------------------------------------------------

when not defined(cpp):
  {.error: "Flambeau requires C++ backend required to use Torch".}

# import ./flambeau/raw/bindings/attensors
#   except # TODO, don't export associated proc either
#     # ArrayRef,
#     TensorOptions,
#     TorchSlice, IndexNone, IndexEllipsis, SomeSlicer, torchSlice
#
# export attensors # TODO, don't export low-level procs and types like C++ slices.

import flambeau/tensors
export tensors

# C++ Standard Library
# ----------------------------------------------------------------

import ./flambeau/raw/cpp/std_cpp
export std_cpp

# Convenience helpers
# ----------------------------------------------------------------

import ./flambeau/sugar/[indexing, interop]
export indexing, interop
