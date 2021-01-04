#!/bin/bash
set -e

cd $(dirname "$0")/../flambeau/libtorch/lib/
echo "Searching $(pwd) for symbol ${1}"

# fd is a parallel alternative to find at https://github.com/sharkdp/fd
fd '.*\.so$' -x bash -c "nm --defined-only {} 2>/dev/null | grep ${1} && echo {}"
