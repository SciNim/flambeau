import ../../flambeau_nn

# Learning XOR function with a neural network.

let bsz = 32 # batch size

# We will create a tensor of size 3200 (100 batches of size 32)
# We create it as int between [0, 1] and convert to bool
var x_train_bool = randInt(0, stopEx=2, bsz*100, 2).to(kBool)

# Let's build our truth labels. We need to apply xor between the 2 columns of the tensors
let y_bool = x_train_bool[_, 0] xor x_train_bool[_,1]

# Convert to float
let x_train = x_train_bool.to(kFloat32)
let y = y_bool.to(kFloat32)

# We will build the following network:
# Input --> Linear(out_features = 3) --> relu --> Linear(out_features = 1)

emitTypes:
  """
  struct XorNet: public torch::nn::Module {
    torch::nn::Linear hidden{nullptr};
    torch::nn::Linear classifier{nullptr};
  };
  """

type XorNet {.pure, importcpp.} = object of Module
  hidden: Linear
  classifier: Linear

proc init(net: var XorNet) =
  net.hidden = net.register_module("hidden", Linear.init(2, 3))
  net.classifier = net.register_module("classifier", Linear.init(3, 1))

proc forward(net: XorNet, x: Tensor): Tensor =
  var x = net.hidden.forward(x).relu()
  return net.classifier.forward(x).squeeze(1)

proc main() =
  var model: XorNet
  model.init()

  # Stochastic Gradient Descent
  var optimizer = SGD.init(
    model.parameters(),
    learning_rate = 0.01
  )

  # Learning loop
  for epoch in 0..5:
    for batch_id in 0 ..< 100:
      # Reset gradients.
      optimizer.zero_grad()

      # minibatch offset in the Tensor
      let offset = batch_id * 32
      let x = x_train[offset ..< offset + 32, _ ]
      let target = y[offset ..< offset + 32]

      # Running input through the network
      let output = model.forward(x)

      # Computing the loss
      var loss = sigmoid_cross_entropy(output, target)

      echo "Epoch is:" & $epoch
      echo "Batch id:" & $batch_id
      echo "Loss is:" & $loss.item(float32)

      # Compute the gradient (i.e. contribution of each parameter to the loss)
      loss.backward()

      # Correct the weights now that we have the gradient information
      optimizer.step()

main()
