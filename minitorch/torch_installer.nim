# Minitorch
# Copyright (c) 2020 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/[asyncdispatch, httpclient,
       strformat, strutils, os],
  zip/zipfiles

type
  Acceleration* = enum
    Cpu = "cpu"
    Cuda92 = "cu92"
    Cuda101 = "cu101"
    Cuda102 = "cu102"

  ABI* = enum
    Cpp = "shared-with-deps"
    Cpp11 = "cxx11-abi-shared-with-deps"

proc getProjectDir(): string {.compileTime.} =
  currentSourcePath.rsplit(DirSep, 1)[0]

proc onProgressChanged(total, progress, speed: BiggestInt) {.async.} =
  echo &"Downloaded {progress} of {total}"
  echo &"Current rate: {speed.float64 / (1000*1000):4.3f} MiBi/s" # TODO the unit is neither MB or Mb or MiBi ???

proc downloadTo(url, targetDir, filename: string) {.async.} =
  var client = newAsyncHttpClient()
  defer: client.close()
  client.onProgressChanged = onProgressChanged
  echo "Starting download of \"", url, '\"'
  echo "Storing temporary into: \"", targetDir, '\"'
  await client.downloadFile(url, targetDir / filename)

proc getUrlAndFilename(version = "1.5.0", accel = Cuda102, abi = Cpp): tuple[url, filename: string] =
  result.filename = "libtorch-"

  when defined(linux):
    result.filename &= &"{abi}-{version}"
    if accel != Cuda102:
      result.filename &= &"%2B{accel}"
    result.filename &= ".zip"
  elif defined(windows):
    doAssert abi == Cpp, "LibTorch for Windows does not support the C++11 ABI"
    result.filename &= &"{win}-{version}-{version}.zip"
  elif defined(osx):
    doAssert accel == Cpu, "LibTorch for MacOS does not support GPU acceleration"
    result.filename &= &"macos-{version}.zip"

  result.url = &"https://download.pytorch.org/libtorch/{accel}/{result.filename}"

proc downloadLibTorch(url, targetDir, filename: string) =
  waitFor url.downloadTo(targetDir, filename)

proc uncompress(targetDir, filename: string, delete = false) =
  var z: ZipArchive
  let tmp = targetDir / filename
  if not z.open(tmp):
    raise newException(IOError, &"Could not open zip file: \"{tmp}\"")
  defer: z.close()
  echo "Decompressing \"", tmp, "\" and storing into \"", targetDir, "\""
  z.extractAll(targetDir)
  echo "Done."
  if delete:
    "Deleting \"", tmp, "\""
  else:
    "Not deleting \"", tmp, "\""

when isMainModule:
  let (url, filename) = getUrlAndFilename()
  uncompress(getProjectDir(), filename)
