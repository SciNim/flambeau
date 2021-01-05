# Flambeau

Flambeau provides Nim bindings to libtorch C++ API.

## Installation

TODO

## Limitations

Compared to Numpy and Arraymancer, Flambeau inherits the following PyTorch limitations:
- No string tensors.
- No support for negative step in tensor slice (`a[::-1]`)
  - https://github.com/pytorch/pytorch/issues/229
  - https://github.com/pytorch/pytorch/issues/604

## License

Licensed and distributed under either of

* MIT license: [LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT

or

* Apache License, Version 2.0, ([LICENSE-APACHEv2](LICENSE-APACHEv2) or http://www.apache.org/licenses/LICENSE-2.0)

at your option. This file may not be copied, modified, or distributed except according to those terms.
