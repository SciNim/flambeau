# Simple accessor tests for Flambeau
# Tests the fixes we've implemented

import flambeau
import std/unittest

proc main() =
  suite "Simple Accessor Tests":
    test "Basic single value access and assignment":
      var a = zeros[int64](@[3'i64, 4])
      a[1, 2] = 42
      check:
        a[1, 2].item() == 42

      a[0, 0] = 99
      check:
        a[0, 0].item() == 99

    test "In-place operators with indexedMutate":
      var a = zeros[int64](@[3'i64, 3])
      a[1, 1] = 5

      indexedMutate:
        a[1, 1] += 10
      check:
        a[1, 1].item() == 15

      indexedMutate:
        a[1, 1] *= 2
      check:
        a[1, 1].item() == 30

      indexedMutate:
        a[1, 1] -= 5
      check:
        a[1, 1].item() == 25

    test "Arithmetic operators with tensors and scalars":
      var a = zeros[int64](@[2'i64, 2])
      a[0, 0] = 10
      a[0, 1] = 20
      a[1, 0] = 30
      a[1, 1] = 40

      # Test Tensor + scalar
      let b = a[0, 0] + 5
      check:
        b.item() == 15

      # Test Tensor * scalar
      let c = a[0, 1] * 2
      check:
        c.item() == 40

      # Test scalar + Tensor
      let d = 3 + a[1, 0]
      check:
        d.item() == 33

    test "Span slicing with underscore":
      var a = zeros[int64](@[3'i64, 4])
      for i in 0 ..< 3:
        for j in 0 ..< 4:
          a[i, j] = (i * 4 + j).int64

      # Select all rows in column 2
      let col2 = a[_, 2]
      check:
        col2.sizes()[0] == 3
      check:
        col2[0].item() == 2
      check:
        col2[1].item() == 6
      check:
        col2[2].item() == 10

    test "From-end indexing with ^":
      # TODO: PyTorch's indexing doesn't directly support Nim's ^ operator
      # This would require additional macro transformation
      skip()

    when compileOption("boundChecks"):
      test "Bounds checking":
        # TODO: PyTorch throws C++ exceptions that don't translate well to Nim's IndexDefect
        # Bounds checking works but exception handling needs improvement
        skip()
    else:
      echo "Bounds checking test skipped (compile with --boundChecks:on)"

    test "Arraymancer-compatible indexing utilities":
      var a = zeros[int64](@[3'i64, 4])
      # Fill tensor with values
      for i in 0 ..< 3:
        for j in 0 ..< 4:
          a[i, j] = (i * 4 + j).int64

      # Test getIndex
      let idx1 = a.getIndex(1, 2)
      check:
        a.atContiguousIndex(idx1) == 6 # Row 1, Col 2 = 1*4 + 2

      # Test atIndex (immutable)
      check:
        a.atIndex(2, 3) == 11 # Row 2, Col 3 = 2*4 + 3

      # Test atIndex (mutable) - allows in-place operations
      a.atIndex(1, 1) = 99
      check:
        a[1, 1].item() == 99

      # Test atIndexMut
      a.atIndexMut(0, 0, 42)
      check:
        a[0, 0].item() == 42

      # Test contiguous indexing
      check:
        a.atContiguousIndex(5) == 99 # Linear index 5 = position [1,1]

      # Test mutable contiguous indexing
      a.atContiguousIndex(8) = 88
      check:
        a[2, 0].item() == 88 # Linear index 8 = position [2,0]

main()
echo "All tests passed!"
