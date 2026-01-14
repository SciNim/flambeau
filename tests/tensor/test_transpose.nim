import flambeau

# Test transpose operations
block:
  echo "\n=== Testing transpose operations ==="

  # Test 1: Basic 2D transpose with t()
  block:
    echo "\n--- Test 1: 2D transpose with t() ---"
    let a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].toTensor()

    echo "Original tensor (2x3):"
    echo a

    let b = a.t()
    echo "\nTransposed with t() (3x2):"
    echo b

    assert b.shape() == @[3'i64, 2'i64]
    assert b[0, 0].item() == 1.0
    assert b[0, 1].item() == 4.0
    assert b[1, 0].item() == 2.0
    assert b[1, 1].item() == 5.0
    echo "✓ Test passed"

  # Test 2: transpose() with specific dimensions
  block:
    echo "\n--- Test 2: transpose() with specific dimensions ---"
    var a = zeros[float32]([2'i64, 3'i64, 4'i64])
    indexedMutate:
      for i in 0 ..< 2:
        for j in 0 ..< 3:
          for k in 0 ..< 4:
            a[i, j, k] = float32(i * 100 + j * 10 + k)

    echo "Original shape: ", a.shape()

    # Transpose dimensions 0 and 2
    let b = a.transpose(0, 2)
    echo "After transpose(0, 2): ", b.shape()
    assert b.shape() == @[4'i64, 3'i64, 2'i64]

    # Verify data moved correctly
    assert b[0, 0, 0].item() == 0.0'f32 # was a[0, 0, 0]
    assert b[1, 0, 0].item() == 1.0'f32 # was a[0, 0, 1]
    assert b[0, 1, 0].item() == 10.0'f32 # was a[0, 1, 0]
    echo "✓ Test passed"

  # Test 3: permute() with arbitrary dimension ordering
  block:
    echo "\n--- Test 3: permute() with arbitrary ordering ---"
    var a = zeros[float32]([2'i64, 3'i64, 4'i64])
    indexedMutate:
      for i in 0 ..< 2:
        for j in 0 ..< 3:
          for k in 0 ..< 4:
            a[i, j, k] = float32(i * 100 + j * 10 + k)

    echo "Original shape: ", a.shape()

    # Permute to [4, 2, 3] - reorder as (2, 0, 1)
    let b = a.permute([2'i64, 0'i64, 1'i64])
    echo "After permute([2, 0, 1]): ", b.shape()
    assert b.shape() == @[4'i64, 2'i64, 3'i64]

    # Verify data moved correctly
    assert b[0, 0, 0].item() == 0.0'f32 # was a[0, 0, 0]
    assert b[1, 0, 0].item() == 1.0'f32 # was a[0, 0, 1]
    assert b[0, 1, 0].item() == 100.0'f32 # was a[1, 0, 0]
    echo "✓ Test passed"

echo "\n✓ All transpose tests passed!"
