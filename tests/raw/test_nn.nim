import ../../flambeau/[flambeau_raw, flambeau_nn]
import unittest

{.experimental: "views".} # TODO

proc main() =
  suite "Functional API":
    test "linear":
      let inputSize = [22'i64, 100] # 100 features in 22 batches
      let outputSize = [22'i64, 10] # 10 features in 22 batches
      let wSize = [10'i64, 100] # outputs 10 features from input of 100 features
      let input = rand(inputSize.asTorchView, kFloat32)
      let w = rand(wSize.asTorchView, kFloat32) # weight matrix
      let b = rand(outputSize.asTorchView, kFloat32) # bias
      let outputWOnly = linear(input, w)
      check outputWOnly.sizes == outputSize.asTorchView
      let outputWb = linear(input, w, b)
      check outputWb.sizes == outputSize.asTorchView
    test "max_pool2d":
      let inputSize = [23'i64, 3, 128, 256] # 3-channel image with size 128x256 in 23 batches
      let kernelSize = [2'i64, 2] # 2x2 kernel will half the spatial dimensions
      let outputSize = [23'i64, 3, 64, 128]
      let input = rand(inputSize.asTorchView, kFloat32)
      let output = max_pool2d(input, kernelSize.asTorchView)
      check output.sizes == outputSize.asTorchView
    test "relu":
      let inputSize = [23'i64, 3, 128, 256]
      var input = zeros(inputSize.asTorchView, kFloat32) + 1.0'f32 
      let output = relu(input)
      check output == input
      input.relu_mut()
      check input == output
    test "log_softmax":
      let inputSize = [24'i64, 10]
      let input = rand(inputSize.asTorchView, kFloat32)
      let output = log_softmax(input, 1)
      check output.sizes == inputSize.asTorchView
    test "dropout":
      let inputSize = [25'i64, 1000]
      let input = rand(inputSize.asTorchView, kFloat32)
      let outputTraining = dropout(input, training=true)
      let outputNotTraining = dropout(input, training=false)
      check outputTraining.sizes == inputSize.asTorchView
      check outputTraining != input
      check outputNotTraining.sizes == inputSize.asTorchView
      check outputNotTraining == input

      var input_mutable = input.clone()
      input_mutable.dropout_mut(training=false)
      check input_mutable == input
      input_mutable.dropout_mut(training=true)
      check input_mutable != input


main()
