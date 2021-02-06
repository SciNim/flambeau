import
  flambeau/flambeau_nn,
  std/[math, strutils, strformat, os]

# Where to find the MNIST dataset.
const kDataRoot = "build/mnist"

# The batch size for training.
const kTrainBatchSize = 64

# The batch size for testing.
const kTestBatchSize = 1000

# The number of epochs to train.
const kNumberOfEpochs = 10

# After how many batches to log a new update with the loss value.
const kLogInterval = 10

# Inline C++
# workaround https://github.com/nim-lang/Nim/issues/16664
# and workaround https://github.com/nim-lang/Nim/issues/4687

emitTypes:
  """
  struct Net: public torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::Conv2d conv2{nullptr};
    torch::nn::Dropout2d conv2_drop{nullptr};
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
  };
  """

type Net {.pure, importcpp.} = object of Module
  conv1: Conv2d
  conv2: Conv2d
  conv2_drop: Dropout2d
  fc1: Linear
  fc2: Linear

proc init(net: var Net) =
  net.conv1 = net.register_module("conv1", Conv2d.init(1, 10, 5))
  net.conv2 = net.register_module("conv2", Conv2d.init(10, 20, 5))
  net.conv2_drop = net.register_module("conv2_drop", Dropout2d.init())
  net.fc1 = net.register_module("fc1", Linear.init(320, 50))
  net.fc2 = net.register_module("fc2", Linear.init(50, 10))

func forward(net: Net, x: Tensor): Tensor =
  var x = net.conv1.forward(x).max_pool2d(2).relu()
  x = net.conv2_drop.forward net.conv2.forward(x)
         .max_pool2d(2)
         .relu()
  x = x.view(-1, 320)
  x = net.fc1.forward(x).relu()
  x = x.dropout(p = 0.5, net.is_training())
  x = net.fc2.forward(x)
  return x.log_softmax(axis = 1)

proc train[DataLoader](
    epoch: int,
    model: var Net,
    device: Device,
    data_loader: DataLoader,
    optimizer: var Optimizer,
    dataset_size: int) =
  model.train()
  for batch_idx, batch in dataloader.pairs():
    let data = batch.data.to(device)
    let targets = batch.target.to(device)
    optimizer.zero_grad()
    var output = model.forward(data)
    var loss = nll_loss(output, targets)
    doAssert loss.item(float64).classify() != fcNan
    loss.backward()
    optimizer.step()

    if batch_idx mod kLogInterval == 0:
      stdout.write "\rTrain Epoch $1 [$2/$3] Loss: $4" % [
        $epoch,
        $(batch_idx * batch.data.size(0)),
        $dataset_size,
        loss.item(float64).formatFloat(precision = 4)
      ]

proc test[DataLoader](
    model: var Net,
    device: Device,
    data_loader: DataLoader,
    dataset_size: int) =
  no_grad_mode:
    model.eval()
    var test_loss = 0.0
    var correct = 0
    for batch in data_loader:
      let data = batch.data.to(device)
      let targets = batch.target.to(device)
      let output = model.forward(data)
      test_loss += nll_loss(
        output, targets, Reduction.Sum
      ).item(float64)
      let pred = output.argmax(1)
      correct += pred.eq(targets).sum().item(int)

    test_loss = test_loss / dataset_size.float64()
    echo &"\nTest set: Average loss: {test_loss:.4f} " &
         &"| Accuracy: {correct.float64() / dataset_size.float64():.3f}"

proc main() =
  Torch.manual_seed(1)
  var device_type: DeviceKind
  if Torch.cuda_is_available():
    echo "CUDA available! Training on GPU."
    device_type = kCuda
  else:
    echo "Training on CPU."
    device_type = kCPU
  let device = Device.init(device_type)

  var model: Net
  model.init()
  model.to(device)

  let train_dataset = mnist(kDataRoot)
                        .map(Normalize[Tensor].init(0.1307, 0.3081))
                        .map(Stack[Example[Tensor, Tensor]].init())
  let train_dataset_size = train_dataset.size().value()
  let train_loader = make_data_loader(
    SequentialSampler,
    train_dataset,
    kTrainBatchSize
  )

  let test_dataset = mnist(kDataRoot, kTest)
                        .map(Normalize[Tensor].init(0.1307, 0.3081))
                        .map(Stack[Example[Tensor, Tensor]].init())
  let test_dataset_size = test_dataset.size().value()
  let test_loader = make_data_loader(
    test_dataset,
    kTestBatchSize
  )

  var optimizer = SGD.init(
    model.parameters(),
    SGDOptions.init(0.01).momentum(0.5)
  )

  for epoch in countup(1, kNumberOfEpochs):
    train(epoch, model, device, train_loader.deref(), optimizer, train_dataset_size)
    test(model, device, test_loader.deref(), test_dataset_size)

main()
