# Flambeau

Flambeau provides Nim bindings to libtorch C++ API and torchvision.

The library is not ready for general use at the moment.

## Installation

The library installation including:
- auto-downloading and deploying of libtorch
- optional auto-compiling of torchvision into a shared library (if requested)


### External dependencies

Initialize submodule :
``git submodule update --init --recursive``

On Ubuntu :
``sudo apt-get install ffmpeg libpng-dev libjpeg-dev libzip-dev``

On OpenSuse :
``sudo zypper install libpng16-devel libjpeg62-devel libzip-devel ffmpeg``

### Installation from git clone

``git clone https://github.com/SciNim/flambeau``

``cd flambeau``

``nimble install`` or ``nimble develop``

Note that install and develop will download and build libtorch so it may take a while.


Torchvision can now be built if desired:

``nimble build_torchvision``

## CUDA support

By default the Torch installation downloaded by this package contains
CUDA support. However, by default Nim packages using Flambeau compile
*without* CUDA support. This is controlled by the `cuda` compile time
option. Compile with:

```sh
nim cpp -d:cuda <foo>
```

to compile with CUDA support.

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
