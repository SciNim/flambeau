import ../../flambeau_nn

# Learning XOR function with a neural network.

let bsz = 32 # batch size

# We will create a tensor of size 3200 (100 batches of size 32)
# We create it as int between [0, 1] and convert to bool
var x_train_bool = randInt(0, stopEx=2, bsz*100, 2).to(kBool)

# Let's build our truth labels. We need to apply xor between the 2 columns of the tensors
let y_bool = x_train_bool[_, 0] xor x_train_bool[_,1]
