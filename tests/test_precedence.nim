import unittest
import flambeau

proc main() =
  suite "Test precedence":
    test "+ and *":
      let a = [[1, 2], [3, 4]].toTensor
      let b = -a
      check b * a + b == [[-2, -6], [-12, -20]].toTensor
      check b * (a + b) == [[0, 0], [0, 0]].toTensor
    test "+ and .abs":
      let a = [[1, 2], [3, 4]].toTensor
      let b = -a
      check a + b.abs == [[2, 4],[6, 8]].toTensor
      check (a + b).abs == [[0, 0], [0, 0]].toTensor

  suite "Tensor Indexing":
    test "Print":
      let t = eye([2, 2, 2].asTorchView)
      echo t
    test "sort, argsort":
      discard
    test "all, any":
      discard

    test "Indexing":
      discard
    test "squeezen unsqueeze":
      discard

  suite "Operations":
    test "add, addmv, addmm:
      discard
    test "matmul, mm, bmm":
      discard
    test "min, max, argmin, argmax":
      discard
    test "sum, prod":
      discard
    test "mean, variance, stddev":
      discard

  suite "Tensor creation":
    test "eye, zeros":
      discard

    test "linspace, logspace, arange":
      discard

  suite "FFT":
    test "fft, fft2, fftn":
      discard
    test "ifft, ifft2, ifftn":
      discard
    test "rfft, rfft2, rfftn":
      discard
    test "irfft, irfft2, irfftn":
      discard


main()
