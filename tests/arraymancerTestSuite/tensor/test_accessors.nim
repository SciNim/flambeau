# Copyright 2017 the Arraymancer contributors
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
import unittest, math
import complex except Complex64, Complex32

{.experimental: "views".} # TODO

proc main() =
  suite "Accessing and setting tensor values":
    test "Accessing and setting a single value":
      let tmpSeq1 = @[2'i64,3,4]
      var a = zeros(tmpSeq1.asTorchView)
      a[1,2,2] = 122
      check: a[1,2,2].item(int) == 122
      let tmpSeq2 = @[3'i64,4]
      var b = zeros(tmpSeq2.asTorchView)
      b[1,2] = 12
      check: b[1,2].item(int) == 12
      b[0,0] = 999
      check: b[0,0].item(int) == 999
      b[2,3] = 111
      check: b[2,3].item(int) == 111

      #var c = zeros[Complex[float64]](@[3,4])
      #c[1,2] = complex64(12.0, 0.0)
      #check: c[1,2].re - 12.0 <= 1e9

    when compileOption("boundChecks") and not defined(openmp):
      test "Out of bounds checking":
        # Fails because there is no out of bounds error raised
        let tmpSeq1 = @[2'i64,3,4]
        var a = zeros(tmpSeq1.asTorchView)
        expect(IndexDefect):
          a[2,0,0] = 200
        let tmpSeq2 = @[3'i64,4]
        var b = zeros(tmpSeq2.asTorchView)
        expect(IndexDefect):
          b[3,4] = 999
        expect(IndexDefect):
          echo b[-1,0] # We don't use discard here because with the C++ backend it is optimized away.
        expect(IndexDefect):
          echo b[0,-2]
    else:
      echo "Bound-checking is disabled or OpenMP is used. The out-of-bounds checking test has been skipped."
#[ not implemented yet
    test "Iterators":
      const
        a = @[1, 2, 3, 4, 5]
        b = @[1, 2, 3]
      var
        vd: seq[seq[int]]
        row: seq[int]
      vd = newSeq[seq[int]]()
      for i, aa in a:
        row = newSeq[int]()
        vd.add(row)
        for j, bb in b:
          vd[i].add(aa^bb)

      let nda_vd = vd.toTensor()

      let expected_seq = @[1,1,1,2,4,8,3,9,27,4,16,64,5,25,125]

      var seq_val: seq[int] = @[]
      for i in nda_vd:
        seq_val.add(i)

      check: seq_val == expected_seq

      var seq_validx: seq[tuple[idx: seq[int], val: int]] = @[]
      for i,j in nda_vd:
        seq_validx.add((i,j))

      check: seq_validx[0] == (@[0,0], 1)
      check: seq_validx[10] == (@[3,1], 16)

      let t_nda = transpose(nda_vd)

      var seq_transpose: seq[tuple[idx: seq[int], val: int]] = @[]
      for i,j in t_nda:
        seq_transpose.add((i,j))

      check: seq_transpose[0] == (@[0,0], 1)
      check: seq_transpose[8] == (@[1,3], 16)
]#
#[
    test "indexing + in-place operator":
      # Fails because a[1, 1] returns an immutable Tensor
      let tempArr = [3'i64,3]
      var a = zeros(tempArr.asTorchView)

      a[1,1] += 10

      a[1,1] *= 20

      check: a == [[0,0,0],[0,200,0],[0,0,0]].toTensor
]#
#[ Not implemented yet
    test "Zipping two tensors":
      let a = [[1,2],[3,4]].toTensor()
      let b = [[5,6],[7,8]].toTensor()

      var res = 0
      for ai, bi in zip(a, b):
        res += ai + bi
      check: res == 36
]#
main()
GC_fullCollect()
