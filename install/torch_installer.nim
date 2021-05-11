# Flambeau
# Copyright (c) 2020 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/[asyncdispatch, httpclient,
     strformat, strutils, os],
  #zippy/ziparchives,
  zip/zipfiles

{.passl: "-lz".}

type
  Acceleration* = enum
    Cpu = "cpu"
    Cuda92 = "cu92"
    Cuda101 = "cu101"
    Cuda102 = "cu102"
    Cuda110 = "cu110"
    Cuda111 = "cu111"

  ABI* = enum
    Cpp = "shared-with-deps"
    Cpp11 = "cxx11-abi-shared-with-deps"

proc getProjectDir(): string {.compileTime.} =
  currentSourcePath.rsplit(DirSep, 1)[0]

proc onProgressChanged(total, progress, speed: BiggestInt) =
  echo &"Downloaded {progress} of {total}"
  echo &"Current rate: {speed.float64 / (1000*1000):4.3f} MiBi/s" # TODO the unit is neither MB or Mb or MiBi ???

proc downloadTo(url, targetDir, filename: string) =
  var client = newHttpClient()
  defer: client.close()
  client.onProgressChanged = onProgressChanged
  echo "Starting download of \"", url, '\"'
  echo "Storing temporary into: \"", targetDir, '\"'
  client.downloadFile(url, targetDir / filename)

proc getUrlAndFilename(version = "1.8.1", accel = Cuda111, abi = Cpp11): tuple[url, filename: string] =
  result.filename = "libtorch-"

  when defined(linux):
    result.filename &= &"{abi}-{version}"
    if accel != Cuda102:
      result.filename &= &"%2B{accel}"
    result.filename &= ".zip"
  elif defined(windows):
    let abi = Cpp
    doAssert abi == Cpp, "LibTorch for Windows does not support the C++11 ABI"
    result.filename &= &"win-{abi}-{version}"
    result.filename &= &"%2B{accel}"
    result.filename &= ".zip"
  elif defined(osx):
    doAssert accel == Cpu, "LibTorch for MacOS does not support GPU acceleration"
    result.filename &= &"macos-{version}.zip"

  result.url = &"https://download.pytorch.org/libtorch/{accel}/{result.filename}"

proc downloadLibTorch(url, targetDir, filename: string) =
  if not fileExists(targetDir / "libtorch.zip"):
    url.downloadTo(targetDir, "libtorch.zip")
  else:
    echo "File is already downloaded"

#[ proc uncompress(targetDir, filename: string, delete = false) =
  let tmpZip = targetDir / filename
  var tmpDir = getTempDir() / "libtorch_temp_download"
  var i = 0
  while dirExists(tmpDir & $i):
    inc(i)
  tmpDir = tmpDir & $i
  echo "Tempdir: ", tmpDir
  let folderName = filename[0 ..< ^4] # remove .zip
  let targetDir = targetDir# / "test"
  #removeDir(tmpDir)
  #createDir(tmpDir)
  echo "Decompressing \"", tmpZip, "\" and storing into \"", targetDir, "\""
  extractAll(tmpZip, tmpDir)
  echo "Extraction done! Now copying from temp"
  removeDir(targetDir / "libtorch") # remove old install
  echo "Removed old folder"
  copyDir(tmpDir, targetDir)
  echo "Copy done!"
  removeDir(tmpDir)
  echo "Done."
  
  if delete:
    echo "Deleting \"", tmpZip, "\""
    removeFile(tmpZip)
  else:
    echo "Not deleting \"", tmpZip, "\"" ]#

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
    echo "Deleting \"", tmp, "\""
    removeFile(tmp)
  else:
    echo "Not deleting \"", tmp, "\""

  # let insPath = targetDir & DirSep & "libtorch"
  # echo "[Important]: Make sure that '" & insPath & DirSep & "lib" & "' is in your LIBRARY_PATH."

when isMainModule:
  let (url, filename) = getUrlAndFilename()
  let target = getProjectDir().parentDir() / "vendor"
  downloadLibTorch(url, target, filename)
  uncompress(target, filename)
