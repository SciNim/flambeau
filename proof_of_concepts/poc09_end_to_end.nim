# This is a port of the C++ end-to-end example
# at https://pytorch.org/cppdocs/frontend.html

import ../flambeau

# Argh, need Linear{nullptr} in the codegen
# so we cheat by inlining C++
#
# type Net {.pure.} = object of Module
#
#   fc1: Linear
#   fc2: Linear
#   fc3: Linear

{.emit:["""
struct Net: public torch::nn::Module {
  torch::nn::Linear fc1{nullptr};
  torch::nn::Linear fc2{nullptr};
  torch::nn::Linear fc3{nullptr};
};
"""].}

type Net{.importcpp.} = object of Module
  fc1: Linear
  fc2: Linear
  fc3: Linear

proc init(T: type Net): Net =
  result.fc1 = result.register_module("fc1", Linear.init(784, 64))
  result.fc2 = result.register_module("fc2", Linear.init(64, 32))
  result.fc3 = result.register_module("fc3", Linear.init(32, 10))

func forward*(net: Net, x: Tensor): Tensor =
  var x = x
  x = net.fc1.forward(x.reshape(x.size(0), 784)).relu()
  x = x.dropout(0.5, training = net.is_training())
  x = net.fc2.forward(x).relu()
  x = net.fc3.forward(x).log_softmax(axis = 1)
  return x

proc main() =
  let net = Net.init() # TODO: make_shared

  let data_loader = make_data_loader(
    mnist("build/mnist").map(Stack[Example[Tensor, Tensor]].init()),
    batch_size = 64
  )

  var optimizer = SGD.init(
    net.parameters(),
    learning_rate = 0.01
  )

main()
