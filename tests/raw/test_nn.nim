import ../../flambeau/[flambeau_raw, flambeau_nn]
import std/unittest

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

    #I get SIGSEGV when running this
    test "nll_loss":
      let inputSize = [26'i64, 10]
      let targetSize = [26'i64]
      let input = zeros(inputSize.asTorchView, kFloat32) + 0.5
      let target = zeros(targetSize.asTorchView, kInt64) + 1
      let loss1 = nll_loss(input, target)
      check loss1.dim == 0

    test "binary_cross_entropy_with_logits":
      let inputSize = [26'i64, 10]
      let input = rand(inputSize.asTorchView, kFloat32)
      let target = rand(inputSize.asTorchView, kFloat32)
      let loss = binary_cross_entropy_with_logits(input, target)
      let loss_sigmoid = sigmoid_cross_entropy(input, target)
      check loss.dim == 0
      check loss.numel == 1
      check loss == loss_sigmoid

  suite "Module API":
    test "Linear":
      let linearOptions = LinearOptions.init(100, 10).bias(false)
      var linear = Linear.init(linearOptions)
      let inputSize = [22'i64, 100] # 100 features in 22 batches
      let outputSize = [22'i64, 10] # 10 features in 22 batches
      let input = rand(inputSize.asTorchView, kFloat32)
      let output = linear.forward(input)
      check output.sizes == outputSize.asTorchView
    test "Conv2d":
      var convOptions = Conv2dOptions.init(32, 64, 3).stride(1).padding(1)
      var conv = Conv2d.init(convOptions)#(32, 64, 3) # Take in 32 channels and outputs 64 channel using a kernel with size 3x3
      let inputSize = [17'i64, 32, 128, 128] # 17 batches of 32 channel images with size 128x128
      let outputSize = [17'i64, 64, 128, 128]
      let input = rand(inputSize.asTorchView, kFloat32)
      let output = conv.forward(input)
      check output.sizes == outputSize.asTorchView
    test "Dropout":
      var dropout = Dropout.init()
      let inputSize = [18'i64, 100]
      let input = rand([18'i64, 100].asTorchView, kFloat32)
      let outputTraining = dropout.forward(input)
      check outputTraining.sizes == inputSize.asTorchView
      check outputTraining != input
      dropout.eval()
      let outputEval = dropout.forward(input)
      check outputEval == input
main()
