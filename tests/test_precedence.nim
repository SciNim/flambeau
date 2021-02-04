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

main()
