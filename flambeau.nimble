# Flambeau
# Copyright (c) 2020 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

packageName   = "flambeau"
version       = "0.0.2"
author        = "Mamy André-Ratsimbazafy"
description   = "A state-of-the-art tensor and deep learning backend on CPU, Nvidia Cuda, AMD HIP, OpenCL, Vulkan, OpenGL"
license       = "MIT or Apache License 2.0"
installDirs   = @["vendor"]

### Dependencies
requires "nim >= 1.4.2"
requires "zip"
requires "cppstl >= 0.3.0"
requires "fusion"

backend = "cpp"

import os
when defined(nimdistros):
  import distros
  foreignDep "libjpeg"
  foreignDep "libpng"
  foreignDep "ffmpeg"
  foreignDep "gtest" # TODO remove this dependency (due to io/decoder/sync_decoder_test.cpp)

### Build
task build_torchvision, "Build the dependency torchvision":
  when defined(windows):
    const libName = "torchvision.dll"
  elif defined(macosx):
    const libName = "libtorchvision.dylib"
  else:
    const libName = "libtorchvision.so"

  const libBuilder = "flambeau/install/torchvision_build.nim"
  switch("out", "vendor/" & libName)
  switch("define", "danger")
  switch("app", "lib")
  switch("noMain")
  switch("gc", "none")
  setCommand "cpp", libBuilder


task test_raw, "Execute RawTensor tests ":
  let skipGcArc= @["./test_nn.nim"]
  withDir "tests" / "raw":
    for fstr in listFiles("."):
      if fstr.endsWith(".nim") and fstr.startsWith("." / "test_"):
        echo "Running ", fstr
        if fstr notin skipGcArc:
          selfExec("cpp -r -d:release " & fstr)
          selfExec("cpp -r --gc:arc -d:release " & fstr)
        else:
          echo "Skipping --gc:arc for ", fstr
          selfExec("cpp -r -d:release " & fstr)

task test_tensor, "Execute Tensor[T] tests ":
  withDir "tests" / "tensor":
    for fstr in listFiles("."):
      if fstr.endsWith(".nim") and fstr.startsWith("." / "test_"):
        echo "Running ", fstr
        selfExec("cpp -r -d:release " & fstr)
        selfExec("cpp -r --gc:arc -d:release " & fstr)

task test, "Execute all tests ":
  testRawTask()
  testTensorTask()

task runExamples, "Run all examples":
  withDir "examples" / "proof_of_concepts":
    for fstr in listFiles("."):
      if fstr.endsWith(".nim") and fstr.startsWith("." / "poc"):
        echo "Running ", fstr
        selfExec("cpp -r -d:release " & fstr)
        selfExec("cpp -r --gc:arc -d:release " & fstr)

task install_libtorch, "Download and install libtorch":
  const libInstaller = "flambeau" / "install" / "torch_installer.nim"
  # Using -b:cpp r avoir creating a local binary
  selfExec("-b:cpp r --skipParentCfg:on " & libInstaller)

task setup, "Setup repo":
  if not dirExists "vendor" / "vision" / "torchvision" / "csrc":
    exec("git submodule update --init --recursive")

  if not dirExists "vendor" / "libtorch":
    install_libtorchTask()

after install:
  setupTask()

after develop:
  setupTask()
