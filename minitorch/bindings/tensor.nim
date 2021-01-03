# Minitorch
# Copyright (c) 2020 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/[strutils, os]

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

# {.link: libPath & "libc10" & libSuffix.}
{.link: libPath & "libtorch" & libSuffix.}

# {.passL: "-lc10".}
{.passL: "-ltorch".}
# {.passL: "-Wl,--copy-dt-needed-entries".}

# {.push dynlib: "libtorch" & libSuffix.}

# Headers
# -----------------------------------------------------------------------

const headersPath = currentSourcePath.rsplit(DirSep, 1)[0] & "/../libtorch/include"
const torchHeadersPath = headersPath / "torch/csrc/api/include"
const torchHeader = torchHeadersPath / "torch/torch.h"

{.passC: "-I" & headersPath.}
{.passC: "-I" & torchHeadersPath.}

{.push header: torchHeader.}

# Types
# -----------------------------------------------------------------------

type
  Tensor* {.importcpp: "torch::Tensor", byref.} = object
  TensorOptions* {.importcpp: "torch::TensorOptions", bycopy.} = object

# TensorOptions
# -----------------------------------------------------------------------

func init*(T: type TensorOptions): TensorOptions {.constructor,importcpp: "torch::TensorOptions".}

# Constructors
# -----------------------------------------------------------------------

func init*(T: type Tensor): Tensor {.constructor,importcpp: "torch::Tensor".}

# Functions.h
# -----------------------------------------------------------------------

func eye*(n: int64): Tensor {.importcpp: "torch::eye(@)".}
