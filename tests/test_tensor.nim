import unittest
import flambeau

{.experimental: "views".} # TODO


proc main() =
  suite "Operator precedence":
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

  suite "Tensor creation":
    test "eye, zeros":
      let t = eye(2, kInt64)
      check t == [[1, 0], [0, 1]].toTensor

    test "linspace, logspace, arange":
      discard

  suite "Tensor utils":
    test "Print":
      let shape = [2'i64, 3, 4]
      let t = rand(shape.asTorchView(), kfloat64)
      echo t

    test "sort, argsort":
      let t = [2, 3, 4, 1, 5, 6].toTensor
      let
        s = t.sort()
        args = t.argsort()

      check s.get(0) == [1, 2, 3, 4, 5, 6].toTensor
      check s.get(1) == args
      check args == [3, 0, 1, 2, 4, 5].toTensor

    test "all, any":
      discard

    test "squeezen unsqueeze":
      discard

  suite "Operations":
    test "add, addmv, addmm":
      discard
    test "matmul, mm, bmm":
      discard
    test "min, max, argmin, argmax":
      discard
    test "sum, prod":
      discard
    test "mean, variance, stddev":
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
