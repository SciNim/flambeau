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

when NimMajor <= 1 and NimMinor < 9:
  {.error: "Flambeau requires Nim 2.0-RC or above (1.9.X)"}

import ./flambeau/tensors
export tensors

# C++ Standard Library
# ----------------------------------------------------------------
import ./flambeau/raw/cpp/std_cpp
export std_cpp

import ./flambeau/raw/bindings/c10
export c10

import ./flambeau/raw/sugar/rawinterop

