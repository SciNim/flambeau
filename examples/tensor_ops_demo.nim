## Non-Trivial Tensor Operations Demo
##
## Demonstrates complex tensor manipulations using Flambeau's
## accessor features and high-level API

import ../flambeau
import std/[math, random, strutils]

proc demoIndexing() =
  echo "=".repeat(60)
  echo "DEMO 1: Advanced Indexing with indexedMutate"
  echo "=".repeat(60)

  var matrix = zeros[float32](@[4'i64, 5])

  # Initialize with sequential values using indexedMutate
  indexedMutate:
    for i in 0 ..< 4:
      for j in 0 ..< 5:
        matrix[i, j] = (i * 5 + j).float32

  echo "\nInitial matrix:"
  echo matrix

  # Complex in-place operations
  indexedMutate:
    # Scale diagonal
    for i in 0 ..< 4:
      matrix[i, i] *= 2.0

    # Add to last column
    for i in 0 ..< 4:
      matrix[i, 4] += 10.0

    # Subtract from first row
    for j in 0 ..< 5:
      matrix[0, j] -= 5.0

  echo "\nAfter complex in-place operations:"
  echo matrix

proc demoMatrixComputation() =
  echo "\n"
  echo "=".repeat(60)
  echo "DEMO 2: Matrix Operations & Broadcasting"
  echo "=".repeat(60)

  # Create two matrices
  var a = zeros[float32](@[3'i64, 3])
  var b = zeros[float32](@[3'i64, 3])

  # Fill with patterns using atIndexMut
  for i in 0 ..< 3:
    for j in 0 ..< 3:
      a.atIndexMut(i, j, (i + 1).float32)
      b.atIndexMut(i, j, (j + 1).float32)

  echo "\nMatrix A:"
  echo a
  echo "\nMatrix B:"
  echo b

  # Element-wise operations
  let sum = a + b
  let diff = a - b
  let prod = a * b # Element-wise product

  echo "\nA + B:"
  echo sum
  echo "\nA - B:"
  echo diff
  echo "\nA * B (element-wise):"
  echo prod

  # Scalar operations
  let scaled = a * 2.5
  let shifted = b + 10.0

  echo "\nA * 2.5:"
  echo scaled
  echo "\nB + 10.0:"
  echo shifted

proc demoReductions() =
  echo "\n"
  echo "=".repeat(60)
  echo "DEMO 3: Reduction Operations"
  echo "=".repeat(60)

  var data = zeros[float32](@[4'i64, 6])

  # Fill with random-like pattern
  randomize(42)
  indexedMutate:
    for i in 0 ..< 4:
      for j in 0 ..< 6:
        data[i, j] = (rand(100).float32 / 10.0'f32)

  echo "\nData matrix:"
  echo data

  # Global reductions
  let total = data.sum().item()
  let avg = data.mean().item()
  let maximum = data.max().item()
  let minimum = data.min().item()

  echo "\nGlobal Statistics:"
  echo "  Sum:  ", total.formatFloat(ffDecimal, 2)
  echo "  Mean: ", avg.formatFloat(ffDecimal, 2)
  echo "  Max:  ", maximum.formatFloat(ffDecimal, 2)
  echo "  Min:  ", minimum.formatFloat(ffDecimal, 2)

  # Axis reductions
  let row_sums = data.sum(@[1'i64]) # Sum along columns
  let col_sums = data.sum(@[0'i64]) # Sum along rows

  echo "\nRow sums (sum along axis 1):"
  echo row_sums
  echo "\nColumn sums (sum along axis 0):"
  echo col_sums

proc demoComplexWorkflow() =
  echo "\n"
  echo "=".repeat(60)
  echo "DEMO 4: Complex Workflow - Linear Transformation"
  echo "=".repeat(60)

  # Simulate a simple linear transformation: y = Wx + b
  # Where W is 4x3, x is Nx3, b is 4, and y is Nx4

  echo "\nCreating transformation parameters..."

  # Weight matrix
  var W = rand[float32](@[4'i64, 3])
  indexedMutate:
    W -= 0.5 # Center around 0
    W *= 2.0 # Scale to [-1, 1]

  # Bias vector
  var b = rand[float32](@[4'i64])
  indexedMutate:
    b -= 0.5

  echo "\nWeight matrix W (4x3):"
  echo W
  echo "\nBias vector b (4):"
  echo b

  # Input data (batch of 5 samples, 3 features each)
  var X = rand[float32](@[5'i64, 3])
  indexedMutate:
    X *= 10.0 # Scale to [0, 10]

  echo "\nInput X (5x3):"
  echo X

  # Apply transformation: Y = X @ W.T + b
  # Since we don't have transpose, we'll reshape and use mm differently
  # For each sample in X, compute W @ x + b
  var Y = zeros[float32](@[5'i64, 4])

  echo "\nApplying linear transformation..."
  for sample_idx in 0 ..< 5:
    # Extract sample (1x3)
    let x_sample = X[sample_idx, _]

    # For each output dimension
    for out_idx in 0 ..< 4:
      # Get weight row (1x3)
      let w_row = W[out_idx, _]

      # Compute dot product + bias using accessors
      var dot_product: float32 = 0.0
      for j in 0 ..< 3:
        dot_product += x_sample[j].item() * w_row[j].item()
      dot_product += b.atContiguousIndex(out_idx)

      # Store result using atIndexMut
      Y.atIndexMut(sample_idx, out_idx, dot_product)

  echo "\nOutput Y (5x4):"
  echo Y

  # Compute output statistics
  echo "\nOutput Statistics per sample:"
  for i in 0 ..< 5:
    let row = Y[i, _]
    let row_mean = row.mean().item()
    let row_max = row.max().item()
    let row_min = row.min().item()
    echo "  Sample ",
      i,
      ": mean=",
      row_mean.formatFloat(ffDecimal, 2),
      " max=",
      row_max.formatFloat(ffDecimal, 2),
      " min=",
      row_min.formatFloat(ffDecimal, 2)

proc main() =
  echo "\n"
  echo "#".repeat(60)
  echo "# Flambeau Advanced Tensor Operations Demo"
  echo "#".repeat(60)

  demoIndexing()
  demoMatrixComputation()
  demoReductions()
  demoComplexWorkflow()

  echo "\n"
  echo "#".repeat(60)
  echo "# All demos completed successfully!"
  echo "#".repeat(60)
  echo ""

when isMainModule:
  main()
