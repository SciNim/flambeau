## Complete Neural Network Example: XOR Problem
##
## This demonstrates a non-trivial use case with:
## - Multi-layer perceptron (2 hidden layers)
## - Backpropagation and gradient descent
## - Training loop with loss monitoring
## - Using Flambeau's accessor features

import ../flambeau
import std/[random, math, strutils]

# Neural Network Architecture
type NeuralNetwork = object
  w1: Tensor[float32] # Input to hidden1: 2x4
  b1: Tensor[float32] # Bias for hidden1: 4
  w2: Tensor[float32] # Hidden1 to hidden2: 4x4
  b2: Tensor[float32] # Bias for hidden2: 4
  w3: Tensor[float32] # Hidden2 to output: 4x1
  b3: Tensor[float32] # Bias for output: 1

proc initNetwork(): NeuralNetwork =
  ## Initialize network with random weights
  # Use uniform random initialization scaled for Xavier-like initialization
  result.w1 = (rand[float32](@[2'i64, 4]) - 0.5) * 2.0 * sqrt(6.0 / 6.0)
  result.b1 = zeros[float32](@[4'i64])

  result.w2 = (rand[float32](@[4'i64, 4]) - 0.5) * 2.0 * sqrt(6.0 / 8.0)
  result.b2 = zeros[float32](@[4'i64])

  result.w3 = (rand[float32](@[4'i64, 1]) - 0.5) * 2.0 * sqrt(6.0 / 5.0)
  result.b3 = zeros[float32](@[1'i64])

proc sigmoid(x: Tensor[float32]): Tensor[float32] =
  ## Sigmoid activation function
  1.0 / (1.0 + exp(-x))

proc sigmoidGrad(x: Tensor[float32]): Tensor[float32] =
  ## Gradient of sigmoid
  let s = sigmoid(x)
  s * (1.0 - s)

proc forward(
    net: NeuralNetwork, x: Tensor[float32]
): tuple[output: Tensor[float32], hidden1: Tensor[float32], hidden2: Tensor[float32], z1, z2, z3: Tensor[float32]] =
  ## Forward pass through the network
  ## Returns activations and pre-activation values for backprop

  # Layer 1: Input -> Hidden1
  let z1 = matmul(x, net.w1) + net.b1
  let hidden1 = sigmoid(z1)

  # Layer 2: Hidden1 -> Hidden2
  let z2 = matmul(hidden1, net.w2) + net.b2
  let hidden2 = sigmoid(z2)

  # Layer 3: Hidden2 -> Output
  let z3 = matmul(hidden2, net.w3) + net.b3
  let output = sigmoid(z3)

  (output, hidden1, hidden2, z1, z2, z3)

proc mse_loss(pred, target: Tensor[float32]): float32 =
  ## Mean squared error loss
  let diff = pred - target
  (diff * diff).mean().item()

proc backward(
    net: var NeuralNetwork,
    x, target: Tensor[float32],
    output, hidden1, hidden2, z1, z2, z3: Tensor[float32],
    learning_rate: float32,
) =
  ## Backpropagation and weight update

  # Output layer gradient
  let dL_doutput = 2.0 * (output - target) / target.numel().float32
  let dL_dz3 = dL_doutput * sigmoidGrad(z3)

  # Gradients for w3 and b3
  let dL_dw3 = matmul(hidden2.transpose(0, 1), dL_dz3)
  let dL_db3 = dL_dz3.sum(@[0'i64])

  # Hidden layer 2 gradient
  let dL_dhidden2 = matmul(dL_dz3, net.w3.transpose(0, 1))
  let dL_dz2 = dL_dhidden2 * sigmoidGrad(z2)

  # Gradients for w2 and b2
  let dL_dw2 = matmul(hidden1.transpose(0, 1), dL_dz2)
  let dL_db2 = dL_dz2.sum(@[0'i64])

  # Hidden layer 1 gradient
  let dL_dhidden1 = matmul(dL_dz2, net.w2.transpose(0, 1))
  let dL_dz1 = dL_dhidden1 * sigmoidGrad(z1)

  # Gradients for w1 and b1
  let dL_dw1 = matmul(x.transpose(0, 1), dL_dz1)
  let dL_db1 = dL_dz1.sum(@[0'i64])

  # Update weights using gradient descent
  indexedMutate:
    net.w1 -= learning_rate * dL_dw1
    net.b1 -= learning_rate * dL_db1
    net.w2 -= learning_rate * dL_dw2
    net.b2 -= learning_rate * dL_db2
    net.w3 -= learning_rate * dL_dw3
    net.b3 -= learning_rate * dL_db3

proc train() =
  echo "Training Neural Network on XOR Problem"
  echo "========================================"

  # XOR dataset
  let x_data = [[0.0'f32, 0.0], [0.0'f32, 1.0], [1.0'f32, 0.0], [1.0'f32, 1.0]]

  let y_data = [
    [0.0'f32], # 0 XOR 0 = 0
    [1.0'f32], # 0 XOR 1 = 1
    [1.0'f32], # 1 XOR 0 = 1
    [0.0'f32], # 1 XOR 1 = 0
  ]

  # Convert to tensors
  var x_train = zeros[float32](@[4'i64, 2])
  var y_train = zeros[float32](@[4'i64, 1])

  indexedMutate:
    for i in 0 ..< 4:
      for j in 0 ..< 2:
        x_train[i, j] = x_data[i][j]
      y_train[i, 0] = y_data[i][0]

  echo "\nTraining Data:"
  echo "X:\n", x_train
  echo "Y:\n", y_train

  # Initialize network
  var net = initNetwork()

  # Training hyperparameters
  const
    epochs = 5000
    learning_rate = 0.5'f32
    print_every = 500

  # Training loop
  echo "\nTraining Progress:"
  echo "------------------"

  for epoch in 0 ..< epochs:
    # Forward pass
    let (output, h1, h2, z1, z2, z3) = net.forward(x_train)

    # Compute loss
    let loss = mse_loss(output, y_train)

    # Backward pass and update
    net.backward(x_train, y_train, output, h1, h2, z1, z2, z3, learning_rate)

    # Print progress
    if epoch mod print_every == 0:
      echo "Epoch ", epoch, " - Loss: ", loss.formatFloat(ffDecimal, 6)

  # Final evaluation
  echo "\nFinal Results:"
  echo "--------------"
  let (final_output, _, _, _, _, _) = net.forward(x_train)

  echo "\nPredictions vs Targets:"
  for i in 0 ..< 4:
    let pred = final_output[i, 0].item()
    let target = y_train[i, 0].item()
    let input1 = x_train[i, 0].item()
    let input2 = x_train[i, 1].item()
    let rounded = if pred > 0.5: 1 else: 0
    echo "Input: [",
      input1.formatFloat(ffDecimal, 1),
      ", ",
      input2.formatFloat(ffDecimal, 1),
      "] -> ",
      "Pred: ",
      pred.formatFloat(ffDecimal, 4),
      " (",
      rounded,
      ")",
      " Target: ",
      target.formatFloat(ffDecimal, 1)

  # Test accuracy
  var correct = 0
  for i in 0 ..< 4:
    let pred = final_output[i, 0].item()
    let target = y_train[i, 0].item()
    if (pred > 0.5 and target > 0.5) or (pred <= 0.5 and target <= 0.5):
      correct += 1

  echo "\nAccuracy: ", correct, "/4 (", (correct * 100 / 4), "%)"

  # Final loss
  let final_loss = mse_loss(final_output, y_train)
  echo "Final Loss: ", final_loss.formatFloat(ffDecimal, 6)

when isMainModule:
  train()
