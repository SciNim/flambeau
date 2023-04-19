import std/unittest
import std/sequtils
import std/strutils
import std/complex
import std/sugar

#import flambeau
include flambeau

{.experimental: "views".} # TODO

proc main() =
  suite "Operator precedence":
    test "= and ==":
      let a = [[1, 2], [3, 4]].toTensor
      let b = a
      check: a == b
    test "+ and *":
      let a = [[1, 2], [3, 4]].toTensor
      let b = -a
      check b * a + b == [[-2, -6], [-12, -20]].toTensor
      check b * (a + b) == [[0, 0], [0, 0]].toTensor
    test "+ and .abs":
      let a = [[1, 2], [3, 4]].toTensor
      let b = -a
      check a + b.abs == [[2, 4], [6, 8]].toTensor
      check (a + b).abs == [[0, 0], [0, 0]].toTensor

  suite "Tensor creation":
    test "eye":
      let t = eye[int](2)
      check t == [[1, 0], [0, 1]].toTensor

    test "zeros":
      let shape = [2'i64, 3]
      let t = zeros[float32](shape)
      check t == [[0.0'f32, 0.0, 0.0], [0.0'f32, 0.0, 0.0]].toTensor

    test "linspace":
      let steps = 120'i64
      let reft = toSeq(0..<120).map(x => float64(x)/float64(steps-1)).toTensor()
      let t = linspace[float64](0.0, 1.0, steps)
      # Max value is 1.0 no need to divide
      let rel_error = mean(t - reft)
      check rel_error.item() <= 1e-12

    test "arange":
      let steps = 130'i64
      let step = 1.0/float64(steps)
      let t = arange[float64](0.0, 1.0, step)
      for i in 0..<130:
        let val = t[i].item()
        let refval: float64 = i.float64 / 130.0
        check (val - refval) < 1e-12

  suite "Tensor utils":
    test "Print":
      let t = [
        [[1.0'f64, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]],
        [[1.0'f64, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]],
      ].toTensor()
      echo t
      var s = strip($t)
      check s.toHex == "54656E736F725B666C6F617436345D0A28312C2E2C2E29203D200A202031202D31202D310A202D31202031202D310A202D31202D312020310A0A28322C2E2C2E29203D200A202031202D31202D310A202D31202031202D310A202D31202D312020310A5B20435055446F75626C65547970657B322C332C337D205D"

    test "sort, argsort":
      let t = [2, 3, 4, 1, 5, 6].toTensor()
      let
        s = sort(t)
        args = argsort(t)
      check s.values == [1, 2, 3, 4, 5, 6].toTensor()
      check s.indices == args
      check args == [3, 0, 1, 2, 4, 5].toTensor().to(int64)

    test "reshape":
      block:
        var tt: Tensor[int] = [[-2, -6], [-12, -20]].toTensor()
        check tt.shape() == [2'i64, 2]
        let tt2 = tt.reshape(@[2'i64, 2, 1])
        check tt2.shape() == [2'i64, 2, 1]

      block:
        var tt: Tensor[int] = [[1, 2, 3, 4], [5, 6, 7, 8]].toTensor()
        var tt3 = tt.reshape([4'i64, 2])
        check tt3.shape() == [4'i64, 2]
        check tt3 == [[1, 2], [3, 4], [5, 6], [7, 8]].toTensor()

    test "Flip, Concat":
      var inttens: Tensor[int] = [[1, 2, 3], [4, 5, 6]].toTensor()
      var fliptens = flip(inttens, [1'i64])
      check fliptens == [[3, 2, 1], [6, 5, 4]].toTensor()
      var catfliptens = concat(fliptens, inttens)
      check catfliptens == [[3, 2, 1], [6, 5, 4], [1, 2, 3], [4, 5, 6]].toTensor()

      # test "all, any":
      #   discard
      #
      # test "squeezen unsqueeze":
      #   discard

  suite "Operations":
    test "add, addmv, addmm":
      discard
    test "matmul, mm, bmm":
      discard

  suite "FFT1D":
    setup:
      let shape = [8'i64]
      var f64input {.used.} = rand[float64](shape)
      var c64input {.used.} = rand[Complex64](shape)

    test "item()":
      # Check item for complex
      let m = c64input[0].item()
      check: $typeof(m) == "Complex64"
      check m.re is float64
      check m.im is float64

    test "fft, ifft":
      # echo c64input
      let fftout = fft(c64input)
      # echo fftout
      let ifftout = ifft(fftout)
      # echo ifftout
      let max_input = max(abs(ifftout)).item()
      # echo max_input
      # Compare abs of Complex values
      var rel_diff = abs(ifftout - c64input) #.to(float64)
                                             # echo rel_diff
      rel_diff /= max_input
      # This isn't a perfect way of checking if Complex number are close enough
      # But it'll do for this simple case
      check mean(rel_diff).item() < 1e-12

    test "rfft, irfft":
      # echo f64input
      let fftout = rfft(f64input)
      # echo fftout
      let ifftout = irfft(fftout)
      # echo ifftout
      let max_input = max(abs(ifftout)).item()
      # echo max_input
      # Compare abs of Complex values
      var rel_diff = abs(ifftout - f64input)
      rel_diff /= max_input
      # echo rel_diff
      # This isn't a perfect way of checking if Complex number are close enough
      # But it'll do for this simple case
      check mean(rel_diff).item() < 1e-12
      # echo mean(rel_diff).item()

  suite "FFT2D":
    setup:
      let shape = [3'i64, 5]
      var f64input {.used.} = rand[float64](shape)
      var c64input {.used.} = rand[Complex64](shape)

    test "fft2, ifft2":
      let fft2out = fft2(c64input)
      # echo fft2out
      let ifft2out = ifft2(fft2out)
      # echo ifft2out
      let max_input = max(abs(ifft2out)).item()
      # Compare abs of Complex values
      var rel_diff = abs(ifft2out - c64input)
      rel_diff /= max_input
      # This isn't a perfect way of checking if Complex number are close enough
      # But it'll do for this simple case
      check mean(rel_diff).item() < 1e-12

  suite "FFTND":
    setup:
      let shape = [3'i64, 4, 5]
      var f64input {.used.} = rand[float64](shape)
      var c64input {.used.} = rand[Complex64](shape)

    test "fftn, ifftn":
      let fftnout = fftn(c64input)
      # echo fftnout
      let ifftnout = ifftn(fftnout)
      # echo ifftnout
      let max_input = max(abs(ifftnout)).item()
      # Compare abs of Complex values
      var rel_diff = abs(ifftnout - c64input)
      rel_diff /= max_input
      # This isn't a perfect way of checking if Complex number are close enough
      # But it'll do for this simple case
      check mean(rel_diff).item() < 1e-12

when isMainModule:
  main()
