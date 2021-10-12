import std/unittest
import std/sequtils
import std/complex
import std/sugar

import ../../flambeau/flambeau_raw

{.experimental: "views".} # TODO

proc main() =
  suite "Operator precedence":
    test "+ and *":
      let a = [[1, 2], [3, 4]].toRawTensor
      let b = -a
      check b * a + b == [[-2, -6], [-12, -20]].toRawTensor
      check b * (a + b) == [[0, 0], [0, 0]].toRawTensor
    test "+ and .abs":
      let a = [[1, 2], [3, 4]].toRawTensor
      let b = -a
      check a + b.abs == [[2, 4], [6, 8]].toRawTensor
      check (a + b).abs == [[0, 0], [0, 0]].toRawTensor

  suite "Tensor creation":
    test "eye":
      let t = eye(2, kInt64)
      check t == [[1, 0], [0, 1]].toRawTensor

    test "zeros":
      let shape = [2'i64, 3]
      let t = zeros(shape.asTorchView(), kFloat32)
      check t == [[0.0'f32, 0.0, 0.0], [0.0'f32, 0.0, 0.0]].toRawTensor

    test "linspace":
      let steps = 120'i64
      let reft = toSeq(0..<120).map(x => float64(x)/float64(steps-1)).toRawTensor()
      let t = linspace(0.0, 1.0, steps, kFloat64)
      # Max value is 1.0 no need to divide
      let rel_error = mean(t - reft)
      check rel_error.item(float64) <= 1e-12

    test "arange":
      let steps = 130'i64
      let step = 1.0/float64(steps)
      let t = arange(0.0, 1.0, step, float64)
      for i in 0..<130:
        let val = t[i].item(float64)
        let refval: float64 = i.float64 / 130.0
        check (val - refval) < 1e-12

    # test "inv":
    #   let t = [[1, 2], [3, 4]].toRawTensor
    #   echo t.inv()

  suite "Tensor utils":
    test "Print":
      let shape = [2'i64, 3, 4]
      let t = rand(shape.asTorchView(), kfloat64)
      echo t

    test "sort, argsort":
      let t = [2, 3, 4, 1, 5, 6].toRawTensor
      let
        s = t.sort()
        args = t.argsort()
      check s.get(0) == [1, 2, 3, 4, 5, 6].toRawTensor()
      check s.get(1) == args
      check args == [3, 0, 1, 2, 4, 5].toRawTensor()

    test "all, any":
      discard

    test "squeezen unsqueeze":
      discard

  suite "Operations":
    test "add, addmv, addmm":
      discard
    test "matmul, mm, bmm":
      discard

  suite "FFT1D":
    setup:
      let shape = [8'i64]
      var f64input {.used.} = rand(shape.asTorchView(), kfloat64)
      var c64input {.used.} = rand(shape.asTorchView(), kComplexF64)

    test "item(Complex64)":
      # Check item for complex
      let m: C10_Complex[float64] = c64input[0].item(Complex64)
      check m.real is float64
      check m.imag is float64

    test "fft, ifft":
      let fftout = fft(c64input)
      # echo fftout
      let ifftout = ifft(fftout)
      # echo ifftout
      let max_input = max(abs(ifftout)).item(float64)
      # Compare abs of Complex values
      var rel_diff = abs(ifftout - c64input)
      rel_diff /= max_input
      # This isn't a perfect way of checking if Complex number are close enough
      # But it'll do for this simple case
      check mean(rel_diff).item(float64) < 1e-12

    test "rfft, irfft":
      let fftout = rfft(f64input)
      # echo fftout
      let ifftout = irfft(fftout)
      # echo ifftout
      let max_input = max(abs(ifftout)).item(float64)
      # Compare abs of Complex values
      var rel_diff = abs(ifftout - f64input)
      rel_diff /= max_input
      # This isn't a perfect way of checking if Complex number are close enough
      # But it'll do for this simple case
      check mean(rel_diff).item(float64) < 1e-12

  suite "FFT2D":
    setup:
      let shape = [3'i64, 5]
      var f64input {.used.} = rand(shape.asTorchView(), kfloat64)
      var c64input {.used.} = rand(shape.asTorchView(), kComplexF64)

    test "fft2, ifft2":
      let fft2out = fft2(c64input)
      # echo fft2out
      let ifft2out = ifft2(fft2out)
      # echo ifft2out
      let max_input = max(abs(ifft2out)).item(float64)
      # Compare abs of Complex values
      var rel_diff = abs(ifft2out - c64input)
      rel_diff /= max_input
      # This isn't a perfect way of checking if Complex number are close enough
      # But it'll do for this simple case
      check mean(rel_diff).item(float64) < 1e-12

  suite "FFTND":
    setup:
      let shape = [3'i64, 4, 5]
      var f64input {.used.} = rand(shape.asTorchView(), kfloat64)
      var c64input {.used.} = rand(shape.asTorchView(), kComplexF64)

    test "fftn, ifftn":
      let fftnout = fftn(c64input)
      # echo fftnout
      let ifftnout = ifftn(fftnout)
      # echo ifftnout
      let max_input = max(abs(ifftnout)).item(float64)
      # Compare abs of Complex values
      var rel_diff = abs(ifftnout - c64input)
      rel_diff /= max_input
      # This isn't a perfect way of checking if Complex number are close enough
      # But it'll do for this simple case
      check mean(rel_diff).item(float64) < 1e-12

main()
