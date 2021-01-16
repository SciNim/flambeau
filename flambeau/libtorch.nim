# Flambeau
# Copyright (c) 2020 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/[strutils, os],
  ./config

# #######################################################################
#
#                          C++ Interop
#
# #######################################################################

# Libraries
# -----------------------------------------------------------------------

const libTorchPath* = currentSourcePath.rsplit(DirSep, 1)[0] & "/../vendor/libtorch"
const librariesPath* = libTorchPath & "/lib"

# TODO: proper build system on "nimble install" (put libraries in .nimble/bin?)
# if the libPath is not in LD_LIBRARY_PATH
# The libraries won't be loaded at runtime

when false: # Static linking
  when defined(windows):
    const libSuffix = ".lib"
    const libPrefix = ""
  elif defined(maxosx):
    #
    const libSuffix = ".a" # MacOS
    const libPrefix = "lib"
  else:
    const libSuffix = ".a" # BSD / Linux
    const libPrefix = "lib"

  {.link: librariesPath & "/" & libPrefix & "c10" & libSuffix.}
  {.link: librariesPath & "/" & libPrefix & "torch_cpu" & libSuffix.}

  when UseCuda:
    {.link: librariesPath & "/" & libPrefix & "torch_cuda" & libSuffix.}
else: # Dynamic linking
  # Standard GCC compatible linker
  {.passL: "-L" & librariesPath & " -lc10 -ltorch_cpu ".}

  when UseCuda:
    {.passL: " -ltorch_cuda ".}

  when not UseGlobalTorch:
    # Link to library in vendor (not for deployment!)!
    when defined(macosx):
      {.passL:"-rpath " & librariesPath.}
    elif defined(posix):
      {.passL:"-Wl,-rpath," & librariesPath.}

    # Look next to the final binary
    # when defined(macosx):
    #   {.passL:"-rpath @loader_path".}
    # elif defined(posix):
    #   {.passL:"-Wl,-rpath,\\$ORIGIN".}

{.push cdecl.}

# Headers
# -----------------------------------------------------------------------

const headersPath* = libTorchPath & "/include"
const torchHeadersPath* = headersPath / "torch/csrc/api/include"
const torchHeader* = torchHeadersPath / "torch/torch.h"

{.passC: "-I" & headersPath.}
{.passC: "-I" & torchHeadersPath.}

{.push header: torchHeader.}

{.passC: "-Wfatal-errors".} # The default "-fmax-errors=3" is unreadable
