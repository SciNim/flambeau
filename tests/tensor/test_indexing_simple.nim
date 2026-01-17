import flambeau

proc main() =
  echo "Testing Arraymancer-compatible indexing functions..."

  # Create a simple 2D tensor
  var a = zeros[int64](@[3'i64, 4])

  # Fill it with values using PyTorch's indexing (which works)
  for i in 0 ..< 3:
    for j in 0 ..< 4:
      a[i, j] = (i * 4 + j).int64

  echo "Tensor created and filled:"
  echo a

  # Test getIndex - compute linear index from coordinates
  echo "\nTesting getIndex..."
  let idx1 = a.getIndex(0, 0)
  echo "getIndex(0, 0) = ", idx1, " (should be 0)"

  let idx2 = a.getIndex(1, 2)
  echo "getIndex(1, 2) = ", idx2, " (should be 6 for row-major: 1*4 + 2)"

  # Test atIndex - read value at coordinates
  echo "\nTesting atIndex (read)..."
  let val1 = a.atIndex(0, 0)
  echo "atIndex(0, 0) = ", val1, " (should be 0)"

  let val2 = a.atIndex(1, 2)
  echo "atIndex(1, 2) = ", val2, " (should be 6)"

  # Test atIndexMut - write value at coordinates
  echo "\nTesting atIndexMut (write)..."
  a.atIndexMut(2, 3, 99)
  let val3 = a[2, 3].item()
  echo "After atIndexMut(2, 3, 99), a[2, 3] = ", val3, " (should be 99)"

  # Test atContiguousIndex
  echo "\nTesting atContiguousIndex..."
  let val4 = a.atContiguousIndex(5)
  echo "atContiguousIndex(5) = ", val4, " (should be 5)"

  # Test mutable contiguous indexing
  a.atContiguousIndex(8) = 88
  let val5 = a[2, 0].item()
  echo "After atContiguousIndex(8) = 88, a[2, 0] = ", val5, " (should be 88)"

  echo "\nAll indexing accessor tests completed!"

when isMainModule:
  main()
