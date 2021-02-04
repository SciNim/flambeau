# This is a port of the C++ end-to-end example
# at https://pytorch.org/cppdocs/frontend.html

import
  flambeau/flambeau_nn,
  std/[enumerate, strformat]

# Inline C++
# workaround https://github.com/nim-lang/Nim/issues/16664
# and workaround https://github.com/nim-lang/Nim/issues/4687

emitTypes:
  """
  struct Net: public torch::nn::Module {
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
    torch::nn::Linear fc3{nullptr};
  };
  """

type Net
  {.pure, importcpp.}
    = object of Module
  fc1: Linear
  fc2: Linear
  fc3: Linear

proc init(net: var Net) =
  # Note: PyTorch Model serialization requires shared_ptr
  net.fc1 = net.register_module("fc1", Linear.init(784, 64))
  net.fc2 = net.register_module("fc2", Linear.init(64, 32))
  net.fc3 = net.register_module("fc3", Linear.init(32, 10))

func forward*(net: Net, x: Tensor): Tensor =
  var x = x
  x = net.fc1.forward(x.reshape(x.size(0), 784)).relu()
  x = x.dropout(0.5, training = net.is_training())
  x = net.fc2.forward(x).relu()
  x = net.fc3.forward(x).log_softmax(axis = 1)
  return x

proc main() =
  let net = make_shared(Net)
  net.init()

  let data_loader = make_data_loader(
    mnist("build/mnist").map(Stack[Example[Tensor, Tensor]].init()),
    batch_size = 64
  )

  var optimizer = SGD.init(
    net.parameters(),
    learning_rate = 0.01
  )

  for epoch in 1 .. 10:
    # Iterate the data loader to yield batches from the dataset.
    for batch_index, batch in data_loader.pairs():
      # Reset gradients.
      optimizer.zero_grad()
      # Execute the model on the input data.
      let prediction = net.forward(batch.data)
      # Compute a loss value to judge the prediction of our model.
      var loss = nll_loss(prediction, batch.target)
      # Compute the gradients of the loss w.r.t. the parameters of our model.
      loss.backward()
      # Update the parameters based on the calculated gradients.
      optimizer.step()
      # output the loss and checkpoint every 100 batches.
      if batch_index mod 100 == 0:
        echo &"Epoch: {epoch} | Batch: {batch_index} | Loss: {loss.item(float32)}"
        # Serialize your model periodically as a checkpoint.
        save(net, "net.pt")

main()
