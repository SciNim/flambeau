type
  Item* {.include: "ordered_dict.h", importcpp: "Item", byref.} = object
  ## A (key, value) pair.

  SequentialSampler* {.include: "data/samplers/sequential.h", importcpp: "SequentialSampler", byref.} = object
  ## A `Sampler` that returns indices sequentially.

  RandomSampler* {.include: "data/samplers/random.h", importcpp: "RandomSampler", byref.} = object
  ## A `Sampler` that returns random indices.

  StreamSampler* {.include: "data/samplers/stream.h", importcpp: "StreamSampler", byref.} = object
  ## A sampler for (potentially infinite) streams of data.

  DistributedRandomSampler* {.include: "data/samplers/distributed.h", importcpp: "DistributedRandomSampler", byref.} = object
  ## Select samples randomly. The sampling order is shuffled at each
  ## `reset()` call.

  DistributedSequentialSampler* {.include: "data/samplers/distributed.h", importcpp: "DistributedSequentialSampler", byref.} = object
  ## Select samples sequentially.

  MNIST* {.include: "data/datasets/mnist.h", importcpp: "MNIST", byref.} = object
  ## The MNIST dataset.

  InputArchive* {.include: "serialize/input-archive.h", importcpp: "InputArchive", byref.} = object
  ## A recursive representation of tensors that can be deserialized from a
  ## file or stream. In most cases, users should not have to interact with
  ## this class, and should instead use `torch::load`.

  OutputArchive* {.include: "serialize/output-archive.h", importcpp: "OutputArchive", byref.} = object

  OptimizerParamState* {.include: "optim/optimizer.h", importcpp: "OptimizerParamState", byref.} = object

  OptimizerOptions* {.include: "optim/optimizer.h", importcpp: "OptimizerOptions", byref.} = object

  OptimizerParamGroup* {.include: "optim/optimizer.h", importcpp: "OptimizerParamGroup", byref.} = object
  ## Stores parameters in the param_group and stores a pointer to the
  ## OptimizerOptions

  Optimizer* {.include: "optim/optimizer.h", importcpp: "Optimizer", byref.} = object

  Module* {.include: "nn/module.h", importcpp: "Module", byref.} = object
  ## The base class for all modules in PyTorch.

  PackedSequence* {.include: "nn/utils/rnn.h", importcpp: "PackedSequence", byref.} = object
  ## Holds the data and list of `batch_sizes` of a packed sequence.

  Conv1dImpl* {.include: "nn/modules/conv.h", importcpp: "Conv1dImpl", byref.} = object
  ## Applies convolution over a 1-D input. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.Conv1d to learn about
  ## the exact behavior of this module.

  Conv2dImpl* {.include: "nn/modules/conv.h", importcpp: "Conv2dImpl", byref.} = object
  ## Applies convolution over a 2-D input. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d to learn about
  ## the exact behavior of this module.

  Conv3dImpl* {.include: "nn/modules/conv.h", importcpp: "Conv3dImpl", byref.} = object
  ## Applies convolution over a 3-D input. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.Conv3d to learn about
  ## the exact behavior of this module.

  ConvTranspose1dImpl* {.include: "nn/modules/conv.h", importcpp: "ConvTranspose1dImpl", byref.} = object
  ## Applies the ConvTranspose1d function. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.ConvTranspose1d to
  ## learn about the exact behavior of this module.

  ConvTranspose2dImpl* {.include: "nn/modules/conv.h", importcpp: "ConvTranspose2dImpl", byref.} = object
  ## Applies the ConvTranspose2d function. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.ConvTranspose2d to
  ## learn about the exact behavior of this module.

  ConvTranspose3dImpl* {.include: "nn/modules/conv.h", importcpp: "ConvTranspose3dImpl", byref.} = object
  ## Applies the ConvTranspose3d function. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.ConvTranspose3d to
  ## learn about the exact behavior of this module.

  AvgPool1dImpl* {.include: "nn/modules/pooling.h", importcpp: "AvgPool1dImpl", byref.} = object
  ## Applies avgpool over a 1-D input. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.AvgPool1d to learn
  ## about the exact behavior of this module.

  AvgPool2dImpl* {.include: "nn/modules/pooling.h", importcpp: "AvgPool2dImpl", byref.} = object
  ## Applies avgpool over a 2-D input. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.AvgPool2d to learn
  ## about the exact behavior of this module.

  AvgPool3dImpl* {.include: "nn/modules/pooling.h", importcpp: "AvgPool3dImpl", byref.} = object
  ## Applies avgpool over a 3-D input. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.AvgPool3d to learn
  ## about the exact behavior of this module.

  MaxPool1dImpl* {.include: "nn/modules/pooling.h", importcpp: "MaxPool1dImpl", byref.} = object
  ## Applies maxpool over a 1-D input. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.MaxPool1d to learn
  ## about the exact behavior of this module.

  MaxPool2dImpl* {.include: "nn/modules/pooling.h", importcpp: "MaxPool2dImpl", byref.} = object
  ## Applies maxpool over a 2-D input. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.MaxPool2d to learn
  ## about the exact behavior of this module.

  MaxPool3dImpl* {.include: "nn/modules/pooling.h", importcpp: "MaxPool3dImpl", byref.} = object
  ## Applies maxpool over a 3-D input. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.MaxPool3d to learn
  ## about the exact behavior of this module.

  AdaptiveMaxPool1dImpl* {.include: "nn/modules/pooling.h", importcpp: "AdaptiveMaxPool1dImpl", byref.} = object
  ## Applies adaptive maxpool over a 1-D input. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.AdaptiveMaxPool1d to
  ## learn about the exact behavior of this module.

  AdaptiveMaxPool2dImpl* {.include: "nn/modules/pooling.h", importcpp: "AdaptiveMaxPool2dImpl", byref.} = object
  ## Applies adaptive maxpool over a 2-D input. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.AdaptiveMaxPool2d to
  ## learn about the exact behavior of this module.

  AdaptiveMaxPool3dImpl* {.include: "nn/modules/pooling.h", importcpp: "AdaptiveMaxPool3dImpl", byref.} = object
  ## Applies adaptive maxpool over a 3-D input. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.AdaptiveMaxPool3d to
  ## learn about the exact behavior of this module.

  AdaptiveAvgPool1dImpl* {.include: "nn/modules/pooling.h", importcpp: "AdaptiveAvgPool1dImpl", byref.} = object
  ## Applies adaptive avgpool over a 1-D input. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.AdaptiveAvgPool1d to
  ## learn about the exact behavior of this module.

  AdaptiveAvgPool2dImpl* {.include: "nn/modules/pooling.h", importcpp: "AdaptiveAvgPool2dImpl", byref.} = object
  ## Applies adaptive avgpool over a 2-D input. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.AdaptiveAvgPool2d to
  ## learn about the exact behavior of this module.

  AdaptiveAvgPool3dImpl* {.include: "nn/modules/pooling.h", importcpp: "AdaptiveAvgPool3dImpl", byref.} = object
  ## Applies adaptive avgpool over a 3-D input. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.AdaptiveAvgPool3d to
  ## learn about the exact behavior of this module.

  MaxUnpool1dImpl* {.include: "nn/modules/pooling.h", importcpp: "MaxUnpool1dImpl", byref.} = object
  ## Applies maxunpool over a 1-D input. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.MaxUnpool1d to learn
  ## about the exact behavior of this module.

  MaxUnpool2dImpl* {.include: "nn/modules/pooling.h", importcpp: "MaxUnpool2dImpl", byref.} = object
  ## Applies maxunpool over a 2-D input. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.MaxUnpool2d to learn
  ## about the exact behavior of this module.

  MaxUnpool3dImpl* {.include: "nn/modules/pooling.h", importcpp: "MaxUnpool3dImpl", byref.} = object
  ## Applies maxunpool over a 3-D input. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.MaxUnpool3d to learn
  ## about the exact behavior of this module.

  FractionalMaxPool2dImpl* {.include: "nn/modules/pooling.h", importcpp: "FractionalMaxPool2dImpl", byref.} = object
  ## Applies fractional maxpool over a 2-D input. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.FractionalMaxPool2d
  ## to learn about the exact behavior of this module.

  FractionalMaxPool3dImpl* {.include: "nn/modules/pooling.h", importcpp: "FractionalMaxPool3dImpl", byref.} = object
  ## Applies fractional maxpool over a 3-D input. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.FractionalMaxPool3d
  ## to learn about the exact behavior of this module.

  LPPool1dImpl* {.include: "nn/modules/pooling.h", importcpp: "LPPool1dImpl", byref.} = object
  ## Applies the LPPool1d function element-wise. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.LPPool1d to learn
  ## about the exact behavior of this module.

  LPPool2dImpl* {.include: "nn/modules/pooling.h", importcpp: "LPPool2dImpl", byref.} = object
  ## Applies the LPPool2d function element-wise. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.LPPool2d to learn
  ## about the exact behavior of this module.

  DropoutImpl* {.include: "nn/modules/dropout.h", importcpp: "DropoutImpl", byref.} = object
  ## Applies dropout over a 1-D input. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.Dropout to learn
  ## about the exact behavior of this module.

  Dropout2dImpl* {.include: "nn/modules/dropout.h", importcpp: "Dropout2dImpl", byref.} = object
  ## Applies dropout over a 2-D input. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.Dropout2d to learn
  ## about the exact behavior of this module.

  Dropout3dImpl* {.include: "nn/modules/dropout.h", importcpp: "Dropout3dImpl", byref.} = object
  ## Applies dropout over a 3-D input. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.Dropout3d to learn
  ## about the exact behavior of this module.

  AlphaDropoutImpl* {.include: "nn/modules/dropout.h", importcpp: "AlphaDropoutImpl", byref.} = object
  ## Applies Alpha Dropout over the input. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.AlphaDropout to learn
  ## about the exact behavior of this module.

  FeatureAlphaDropoutImpl* {.include: "nn/modules/dropout.h", importcpp: "FeatureAlphaDropoutImpl", byref.} = object
  ## See the documentation for `torch::nn::FeatureAlphaDropoutOptions`
  ## class to learn what constructor arguments are supported for this
  ## module.

  Embedding* {.include: "nn/modules/embedding.h", importcpp: "Embedding", byref.} = object
  ## A `ModuleHolder` subclass for `EmbeddingImpl`. See the documentation
  ## for `EmbeddingImpl` class to learn what methods it provides, and
  ## examples of how to use `Embedding` with `torch::nn::EmbeddingOptions`.
  ## See the documentation for `ModuleHolder` to learn about PyTorch's
  ## module storage semantics.

  EmbeddingBag* {.include: "nn/modules/embedding.h", importcpp: "EmbeddingBag", byref.} = object
  ## A `ModuleHolder` subclass for `EmbeddingBagImpl`. See the
  ## documentation for `EmbeddingBagImpl` class to learn what methods it
  ## provides, and examples of how to use `EmbeddingBag` with
  ## `torch::nn::EmbeddingBagOptions`. See the documentation for
  ## `ModuleHolder` to learn about PyTorch's module storage semantics.

  RNNImpl* {.include: "nn/modules/rnn.h", importcpp: "RNNImpl", byref.} = object
  ## A multi-layer Elman RNN module with Tanh or ReLU activation. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.RNN to learn about
  ## the exact behavior of this module.

  LSTMImpl* {.include: "nn/modules/rnn.h", importcpp: "LSTMImpl", byref.} = object
  ## A multi-layer long-short-term-memory (LSTM) module. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.LSTM to learn about
  ## the exact behavior of this module.

  GRUImpl* {.include: "nn/modules/rnn.h", importcpp: "GRUImpl", byref.} = object
  ## A multi-layer gated recurrent unit (GRU) module. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.GRU to learn about
  ## the exact behavior of this module.

  RNNCellImpl* {.include: "nn/modules/rnn.h", importcpp: "RNNCellImpl", byref.} = object
  ## An Elman RNN cell with tanh or ReLU non-linearity. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.RNNCell to learn
  ## about the exact behavior of this module.

  LSTMCellImpl* {.include: "nn/modules/rnn.h", importcpp: "LSTMCellImpl", byref.} = object
  ## A long short-term memory (LSTM) cell. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.LSTMCell to learn
  ## about the exact behavior of this module.

  GRUCellImpl* {.include: "nn/modules/rnn.h", importcpp: "GRUCellImpl", byref.} = object
  ## A gated recurrent unit (GRU) cell. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.GRUCell to learn
  ## about the exact behavior of this module.

  ELUImpl* {.include: "nn/modules/activation.h", importcpp: "ELUImpl", byref.} = object
  ## Applies elu over a given input. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.ELU to learn about
  ## the exact behavior of this module.

  SELUImpl* {.include: "nn/modules/activation.h", importcpp: "SELUImpl", byref.} = object
  ## Applies the selu function element-wise. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.SELU to learn about
  ## the exact behavior of this module.

  HardshrinkImpl* {.include: "nn/modules/activation.h", importcpp: "HardshrinkImpl", byref.} = object
  ## Applies the hard shrinkage function element-wise. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.Hardshrink to learn
  ## about the exact behavior of this module.

  HardtanhImpl* {.include: "nn/modules/activation.h", importcpp: "HardtanhImpl", byref.} = object
  ## Applies the HardTanh function element-wise. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.Hardtanh to learn
  ## about the exact behavior of this module.

  LeakyReLUImpl* {.include: "nn/modules/activation.h", importcpp: "LeakyReLUImpl", byref.} = object
  ## Applies the LeakyReLU function element-wise. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.LeakyReLU to learn
  ## about the exact behavior of this module.

  LogSigmoidImpl* {.include: "nn/modules/activation.h", importcpp: "LogSigmoidImpl", byref.} = object
  ## Applies the LogSigmoid function element-wise. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.LogSigmoid to learn
  ## about the exact behavior of this module.

  SoftmaxImpl* {.include: "nn/modules/activation.h", importcpp: "SoftmaxImpl", byref.} = object
  ## Applies the Softmax function. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.Softmax to learn
  ## about the exact behavior of this module.

  SoftminImpl* {.include: "nn/modules/activation.h", importcpp: "SoftminImpl", byref.} = object
  ## Applies the Softmin function element-wise. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.Softmin to learn
  ## about the exact behavior of this module.

  LogSoftmaxImpl* {.include: "nn/modules/activation.h", importcpp: "LogSoftmaxImpl", byref.} = object
  ## Applies the LogSoftmax function element-wise. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.LogSoftmax to learn
  ## about the exact behavior of this module.

  Softmax2dImpl* {.include: "nn/modules/activation.h", importcpp: "Softmax2dImpl", byref.} = object
  ## Applies the Softmax2d function element-wise. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.Softmax2d to learn
  ## about the exact behavior of this module.

  PReLUImpl* {.include: "nn/modules/activation.h", importcpp: "PReLUImpl", byref.} = object
  ## Applies the PReLU function element-wise. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.PReLU to learn about
  ## the exact behavior of this module.

  ReLUImpl* {.include: "nn/modules/activation.h", importcpp: "ReLUImpl", byref.} = object
  ## Applies the ReLU function element-wise. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.ReLU to learn about
  ## the exact behavior of this module.

  ReLU6Impl* {.include: "nn/modules/activation.h", importcpp: "ReLU6Impl", byref.} = object
  ## Applies the ReLU6 function element-wise. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.ReLU6 to learn about
  ## the exact behavior of this module.

  RReLUImpl* {.include: "nn/modules/activation.h", importcpp: "RReLUImpl", byref.} = object
  ## Applies the RReLU function element-wise. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.RReLU to learn about
  ## the exact behavior of this module.

  CELUImpl* {.include: "nn/modules/activation.h", importcpp: "CELUImpl", byref.} = object
  ## Applies celu over a given input. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.CELU to learn about
  ## the exact behavior of this module.

  GLUImpl* {.include: "nn/modules/activation.h", importcpp: "GLUImpl", byref.} = object
  ## Applies glu over a given input. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.GLU to learn about
  ## the exact behavior of this module.

  GELUImpl* {.include: "nn/modules/activation.h", importcpp: "GELUImpl", byref.} = object
  ## Applies gelu over a given input. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.GELU to learn about
  ## the exact behavior of this module.

  SiLUImpl* {.include: "nn/modules/activation.h", importcpp: "SiLUImpl", byref.} = object
  ## Applies silu over a given input. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.SiLU to learn about
  ## the exact behavior of this module.

  SigmoidImpl* {.include: "nn/modules/activation.h", importcpp: "SigmoidImpl", byref.} = object
  ## Applies sigmoid over a given input. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.Sigmoid to learn
  ## about the exact behavior of this module.

  SoftplusImpl* {.include: "nn/modules/activation.h", importcpp: "SoftplusImpl", byref.} = object
  ## Applies softplus over a given input. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.Softplus to learn
  ## about the exact behavior of this module.

  SoftshrinkImpl* {.include: "nn/modules/activation.h", importcpp: "SoftshrinkImpl", byref.} = object
  ## Applies the soft shrinkage function element-wise. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.Softshrink to learn
  ## about the exact behavior of this module.

  SoftsignImpl* {.include: "nn/modules/activation.h", importcpp: "SoftsignImpl", byref.} = object
  ## Applies Softsign over a given input. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.Softsign to learn
  ## about the exact behavior of this module.

  TanhImpl* {.include: "nn/modules/activation.h", importcpp: "TanhImpl", byref.} = object
  ## Applies Tanh over a given input. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.Tanh to learn about
  ## the exact behavior of this module.

  TanhshrinkImpl* {.include: "nn/modules/activation.h", importcpp: "TanhshrinkImpl", byref.} = object
  ## Applies Tanhshrink over a given input. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.Tanhshrink to learn
  ## about the exact behavior of this module.

  ThresholdImpl* {.include: "nn/modules/activation.h", importcpp: "ThresholdImpl", byref.} = object
  ## Applies the Threshold function element-wise. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.Threshold to learn
  ## about the exact behavior of this module.

  MultiheadAttentionImpl* {.include: "nn/modules/activation.h", importcpp: "MultiheadAttentionImpl", byref.} = object
  ## Applies the MultiheadAttention function element-wise. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.MultiheadAttention to
  ## learn about the exact behavior of this module.

  UpsampleImpl* {.include: "nn/modules/upsampling.h", importcpp: "UpsampleImpl", byref.} = object
  ## Upsamples a given multi-channel 1D (temporal), 2D (spatial) or 3D
  ## (volumetric) data. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.Upsample to learn
  ## about the exact behavior of this module.

  CrossMapLRN2d* {.include: "nn/modules/_functions.h", importcpp: "CrossMapLRN2d", byref.} = object

  ReflectionPad1dImpl* {.include: "nn/modules/padding.h", importcpp: "ReflectionPad1dImpl", byref.} = object
  ## Applies ReflectionPad over a 1-D input. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.ReflectionPad1d to
  ## learn about the exact behavior of this module.

  ReflectionPad2dImpl* {.include: "nn/modules/padding.h", importcpp: "ReflectionPad2dImpl", byref.} = object
  ## Applies ReflectionPad over a 2-D input. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.ReflectionPad2d to
  ## learn about the exact behavior of this module.

  ReplicationPad1dImpl* {.include: "nn/modules/padding.h", importcpp: "ReplicationPad1dImpl", byref.} = object
  ## Applies ReplicationPad over a 1-D input. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.ReplicationPad1d to
  ## learn about the exact behavior of this module.

  ReplicationPad2dImpl* {.include: "nn/modules/padding.h", importcpp: "ReplicationPad2dImpl", byref.} = object
  ## Applies ReplicationPad over a 2-D input. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.ReplicationPad2d to
  ## learn about the exact behavior of this module.

  ReplicationPad3dImpl* {.include: "nn/modules/padding.h", importcpp: "ReplicationPad3dImpl", byref.} = object
  ## Applies ReplicationPad over a 3-D input. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.ReplicationPad3d to
  ## learn about the exact behavior of this module.

  ZeroPad2dImpl* {.include: "nn/modules/padding.h", importcpp: "ZeroPad2dImpl", byref.} = object
  ## Applies ZeroPad over a 2-D input. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.ZeroPad2d to learn
  ## about the exact behavior of this module.

  ConstantPad1dImpl* {.include: "nn/modules/padding.h", importcpp: "ConstantPad1dImpl", byref.} = object
  ## Applies ConstantPad over a 1-D input. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.ConstantPad1d to
  ## learn about the exact behavior of this module.

  ConstantPad2dImpl* {.include: "nn/modules/padding.h", importcpp: "ConstantPad2dImpl", byref.} = object
  ## Applies ConstantPad over a 2-D input. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.ConstantPad2d to
  ## learn about the exact behavior of this module.

  ConstantPad3dImpl* {.include: "nn/modules/padding.h", importcpp: "ConstantPad3dImpl", byref.} = object
  ## Applies ConstantPad over a 3-D input. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.ConstantPad3d to
  ## learn about the exact behavior of this module.

  CosineSimilarityImpl* {.include: "nn/modules/distance.h", importcpp: "CosineSimilarityImpl", byref.} = object
  ## Returns the cosine similarity between :math:`x_1` and :math:`x_2`,
  ## computed along `dim`. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.CosineSimilarity to
  ## learn about the exact behavior of this module.

  PairwiseDistanceImpl* {.include: "nn/modules/distance.h", importcpp: "PairwiseDistanceImpl", byref.} = object
  ## Returns the batchwise pairwise distance between vectors :math:`v_1`,
  ## :math:`v_2` using the p-norm. See
  ## https://pytorch.org/docs/master/nn.html#torch.nn.PairwiseDistance to
  ## learn about the exact behavior of this module.

  ModuleListImpl* {.include: "nn/modules/container/modulelist.h", importcpp: "ModuleListImpl", byref.} = object
  ## A list of `Module`s that registers its elements.

  SequentialImpl* {.include: "nn/modules/container/sequential.h", importcpp: "SequentialImpl", byref.} = object
  ## A list of `Module`s that acts as a `Module` itself.

  Sequential* {.include: "nn/modules/container/sequential.h", importcpp: "Sequential", byref.} = object
  ## A `ModuleHolder` subclass for `SequentialImpl`. See the documentation
  ## for `SequentialImpl` class to learn what methods it provides, or the
  ## documentation for `ModuleHolder` to learn about PyTorch's module
  ## storage semantics.

  ParameterDictImpl* {.include: "nn/modules/container/parameterdict.h", importcpp: "ParameterDictImpl", byref.} = object

  NamedAnyModule* {.include: "nn/modules/container/named_any.h", importcpp: "NamedAnyModule", byref.} = object
  ## Stores a type erased `Module` with name.

  AnyModule* {.include: "nn/modules/container/any.h", importcpp: "AnyModule", byref.} = object
  ## Stores a type erased `Module`.

  AnyValue* {.include: "nn/modules/container/any_value.h", importcpp: "AnyValue", byref.} = object
  ## An implementation of `std::any` which stores a type erased object,
  ## whose concrete value can be retrieved at runtime by checking if the
  ## `typeid()` of a requested type matches the `typeid()` of the object
  ## stored.

  ParameterListImpl* {.include: "nn/modules/container/parameterlist.h", importcpp: "ParameterListImpl", byref.} = object

  FunctionalImpl* {.include: "nn/modules/container/functional.h", importcpp: "FunctionalImpl", byref.} = object
  ## Wraps a function in a `Module`.

  conv_padding_mode_t* {.include: "nn/options/conv.h", importcpp: "conv_padding_mode_t".} = cint
  mode_t* {.include: "nn/options/vision.h", importcpp: "mode_t".} = cint
  padding_mode_t* {.include: "nn/options/vision.h", importcpp: "padding_mode_t".} = cint
  EmbeddingBagMode* {.include: "nn/options/embedding.h", importcpp: "EmbeddingBagMode".} = cint
  reduction_t* {.include: "nn/options/loss.h", importcpp: "reduction_t".} = cint
  reduction_t* {.include: "nn/options/loss.h", importcpp: "reduction_t".} = cint
  reduction_t* {.include: "nn/options/loss.h", importcpp: "reduction_t".} = cint
  reduction_t* {.include: "nn/options/loss.h", importcpp: "reduction_t".} = cint
  reduction_t* {.include: "nn/options/loss.h", importcpp: "reduction_t".} = cint
  reduction_t* {.include: "nn/options/loss.h", importcpp: "reduction_t".} = cint
  reduction_t* {.include: "nn/options/loss.h", importcpp: "reduction_t".} = cint
  reduction_t* {.include: "nn/options/loss.h", importcpp: "reduction_t".} = cint
  reduction_t* {.include: "nn/options/loss.h", importcpp: "reduction_t".} = cint
  reduction_t* {.include: "nn/options/loss.h", importcpp: "reduction_t".} = cint
  reduction_t* {.include: "nn/options/loss.h", importcpp: "reduction_t".} = cint
  reduction_t* {.include: "nn/options/loss.h", importcpp: "reduction_t".} = cint
  reduction_t* {.include: "nn/options/loss.h", importcpp: "reduction_t".} = cint
  reduction_t* {.include: "nn/options/loss.h", importcpp: "reduction_t".} = cint
  reduction_t* {.include: "nn/options/loss.h", importcpp: "reduction_t".} = cint
  reduction_t* {.include: "nn/options/loss.h", importcpp: "reduction_t".} = cint
  reduction_t* {.include: "nn/options/loss.h", importcpp: "reduction_t".} = cint
  reduction_t* {.include: "nn/options/loss.h", importcpp: "reduction_t".} = cint
  rnn_options_base_mode_t* {.include: "nn/options/rnn.h", importcpp: "rnn_options_base_mode_t".} = cint
  nonlinearity_t* {.include: "nn/options/rnn.h", importcpp: "nonlinearity_t".} = cint
  nonlinearity_t* {.include: "nn/options/rnn.h", importcpp: "nonlinearity_t".} = cint
  namedshape_t* {.include: "nn/options/linear.h", importcpp: "namedshape_t".} = cint
  mode_t* {.include: "nn/options/upsampling.h", importcpp: "mode_t".} = cint
  mode_t* {.include: "nn/options/upsampling.h", importcpp: "mode_t".} = cint
  mode_t* {.include: "nn/options/padding.h", importcpp: "mode_t".} = cint
include "optim.h"
include "autograd.h"
include "enum.h"
include "expanding_array.h"
include "all.h"
include "arg.h"
include "linalg.h"
include "jit.h"
include "python.h"
include "torch.h"
include "data.h"
include "serialize.h"
include "types.h"
include "utils.h"
include "cuda.h"
include "fft.h"
include "ordered_dict.h"
include "nn.h"
include "detail/TensorDataContainer.h"
include "detail/static.h"
include "data/dataloader.h"
include "data/dataloader_options.h"
include "data/samplers.h"
include "data/worker_exception.h"
include "data/transforms.h"
include "data/example.h"
include "data/iterator.h"
include "data/datasets.h"
include "data/samplers/custom_batch_request.h"
include "data/samplers/sequential.h"
include "data/samplers/random.h"
include "data/samplers/serialize.h"
include "data/samplers/stream.h"
include "data/samplers/base.h"
include "data/samplers/distributed.h"
include "data/transforms/collate.h"
include "data/transforms/lambda.h"
include "data/transforms/tensor.h"
include "data/transforms/stack.h"
include "data/transforms/base.h"
include "data/datasets/mnist.h"
include "data/datasets/chunk.h"
include "data/datasets/tensor.h"
include "data/datasets/shared.h"
include "data/datasets/map.h"
include "data/datasets/stateful.h"
include "data/datasets/base.h"
include "data/detail/queue.h"
include "data/detail/data_shuttle.h"
include "data/detail/sequencers.h"
include "data/dataloader/stateless.h"
include "data/dataloader/stateful.h"
include "data/dataloader/base.h"
include "serialize/archive.h"
include "serialize/input-archive.h"
include "serialize/tensor.h"
include "serialize/output-archive.h"
include "optim/adam.h"
include "optim/adamw.h"
include "optim/adagrad.h"
include "optim/lbfgs.h"
include "optim/rmsprop.h"
include "optim/sgd.h"
include "optim/optimizer.h"
include "optim/serialize.h"
include "nn/module.h"
include "nn/options.h"
include "nn/pimpl-inl.h"
include "nn/init.h"
include "nn/cloneable.h"
include "nn/pimpl.h"
include "nn/utils.h"
include "nn/modules.h"
include "nn/functional.h"
include "nn/parallel/data_parallel.h"
include "nn/utils/rnn.h"
include "nn/utils/convert_parameters.h"
include "nn/utils/clip_grad.h"
include "nn/modules/conv.h"
include "nn/modules/transformer.h"
include "nn/modules/pixelshuffle.h"
include "nn/modules/adaptive.h"
include "nn/modules/pooling.h"
include "nn/modules/instancenorm.h"
include "nn/modules/dropout.h"
include "nn/modules/embedding.h"
include "nn/modules/loss.h"
include "nn/modules/rnn.h"
include "nn/modules/activation.h"
include "nn/modules/linear.h"
include "nn/modules/upsampling.h"
include "nn/modules/_functions.h"
include "nn/modules/batchnorm.h"
include "nn/modules/common.h"
include "nn/modules/transformerlayer.h"
include "nn/modules/utils.h"
include "nn/modules/normalization.h"
include "nn/modules/padding.h"
include "nn/modules/transformercoder.h"
include "nn/modules/fold.h"
include "nn/modules/distance.h"
include "nn/modules/container/modulelist.h"
include "nn/modules/container/sequential.h"
include "nn/modules/container/parameterdict.h"
include "nn/modules/container/named_any.h"
include "nn/modules/container/any.h"
include "nn/modules/container/any_value.h"
include "nn/modules/container/any_module_holder.h"
include "nn/modules/container/parameterlist.h"
include "nn/modules/container/functional.h"
include "nn/functional/conv.h"
include "nn/functional/vision.h"
include "nn/functional/pixelshuffle.h"
include "nn/functional/pooling.h"
include "nn/functional/instancenorm.h"
include "nn/functional/dropout.h"
include "nn/functional/embedding.h"
include "nn/functional/loss.h"
include "nn/functional/activation.h"
include "nn/functional/linear.h"
include "nn/functional/upsampling.h"
include "nn/functional/batchnorm.h"
include "nn/functional/normalization.h"
include "nn/functional/padding.h"
include "nn/functional/fold.h"
include "nn/functional/distance.h"
include "nn/options/conv.h"
include "nn/options/vision.h"
include "nn/options/transformer.h"
include "nn/options/pixelshuffle.h"
include "nn/options/adaptive.h"
include "nn/options/pooling.h"
include "nn/options/instancenorm.h"
include "nn/options/dropout.h"
include "nn/options/embedding.h"
include "nn/options/loss.h"
include "nn/options/rnn.h"
include "nn/options/activation.h"
include "nn/options/linear.h"
include "nn/options/upsampling.h"
include "nn/options/batchnorm.h"
include "nn/options/transformerlayer.h"
include "nn/options/normalization.h"
include "nn/options/padding.h"
include "nn/options/transformercoder.h"
include "nn/options/fold.h"
include "nn/options/distance.h"
