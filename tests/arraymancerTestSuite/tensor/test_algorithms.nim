# Copyright 2020 the Arraymancer contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import flambeau, ../testutils
import unittest

suite "[Core] Testing algorithm functions":
  #[ Can't find an inplace version of sort
  test "Sort":
    block:
      var t = @[4, 2, 7, 3, 1].toTensor()
      t.sort()
      check t == @[1, 2, 3, 4, 7].toTensor()

    block:
      var t = @[4, 2, 7, 3, 1].toTensor()
      t.sort(order = SortOrder.Descending)
      check t == @[7, 4, 3, 2, 1].toTensor()
  ]#
  test "Sorted":
    # `sort` is libtorch's equivilent of Nim's `sorted`
    block:
      var t = @[4, 2, 7, 3, 1].toTensor()
      let tmp = t.sort
      check tmp.get(0) == @[1, 2, 3, 4, 7].toTensor()

    block:
      var t = @[4, 2, 7, 3, 1].toTensor()
      let tmp = t.sort(descending = true)
      check tmp.get(0) == @[7, 4, 3, 2, 1].toTensor()

  test "Argsort":
    block:
      let t = @[4, 2, 7, 3, 1].toTensor()
      let exp = @[4, 1, 3, 0, 2].toTensor()
      let idxSorted = t.argsort()
      check idxSorted == exp
      check t[idxSorted] == @[1, 2, 3, 4, 7].toTensor()

    block:
      let t = @[4, 2, 7, 3, 1].toTensor()
      let exp = @[2, 0, 3, 1, 4].toTensor()
      let idxSorted = t.argsort(descending = true)
      check idxSorted == exp
      check t[idxSorted] == @[7, 4, 3, 2, 1].toTensor()
