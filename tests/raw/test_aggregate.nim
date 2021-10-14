# Copyright 2021 the Flambeau contributors
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

import flambeau/flambeau_raw
import unittest

{.experimental: "views".} # TODO

proc main() =
  suite "[Core] Testing aggregation functions":
    let t = [[0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [9, 10, 11]].toRawTensor()
    let t_c = t.to(kComplexF64)

    test "Sum":
      check: t.sum.item(int64) == 66
      #check: t_c.sum.item(CppComplex[float64]).toNimComplex == complex(66'f64)
      let row_sum = [[18, 22, 26]].toRawTensor()
      let col_sum = [[3],
                    [12],
                    [21],
                    [30]].toRawTensor()
      check: t.sum(axis = 0, keepDim = true) == row_sum
      check: t.sum(axis = 1, keepDim = true) == col_sum
      check: t_c.sum(axis = 0, keepDim = true) == row_sum.to(kComplexF64)
      check: t_c.sum(axis = 1, keepDim = true) == col_sum.to(kComplexF64)

    test "Mean":
      check: t.to(kFloat64).mean.item(float64) == 5.5 # Note: may fail due to float rounding
      #check: t_c.mean.item(Complex[float64]) == complex(5.5) # Note: may fail due to float rounding

      let row_mean = [[4.5, 5.5, 6.5]].toRawTensor()
      let col_mean = [[1.0],
                      [4.0],
                      [7.0],
                      [10.0]].toRawTensor()
      check: t.to(kFloat64).mean(axis = 0, keepDim = true) == row_mean
      check: t.to(kFloat64).mean(axis = 1, keepDim = true) == col_mean
      check: t_c.mean(axis = 0, keepDim = true) == row_mean.to(kComplexF64)
      check: t_c.mean(axis = 1, keepDim = true) == col_mean.to(kComplexF64)

    test "Product":
      let a = [[1, 2, 4], [8, 16, 32]].toRawTensor()
      let a_c = a.to(kComplexF64)
      check: t.prod().item(int64) == 0
      check: a.prod().item(int64) == 32768
      check: a.to(kFloat64).prod().item(float64) == 32768.0
      check: a.prod(0, keepDim = true) == [[8, 32, 128]].toRawTensor()
      check: a.prod(1, keepDim = true) == [[8], [4096]].toRawTensor()
      #check: t_c.prod().item(Complex[float64]) == complex(0.0)
      #check: a_c.prod().item(Complex[float64]) == complex(32768.0)
      check: a_c.prod(0, keepDim = true) == [[8, 32, 128]].toRawTensor().to(kComplexF64)
      check: a_c.prod(1, keepDim = true) == [[8], [4096]].toRawTensor().to(kComplexF64)

    test "Min":
      let a = [2, -1, 3, -3, 5, 0].toRawTensor()
      check: a.min().item(int64) == -3
      check: a.to(kFloat64).min().item(float64) == -3.0f

      let b = [[1, 2, 3, -4], [0, 4, -2, 5]].toRawTensor()
      check: b.min(0, keepDim = true).get(0) == [[0, 2, -2, -4]].toRawTensor()
      check: b.min(1, keepDim = true).get(0) == [[-4], [-2]].toRawTensor()
      check: b.to(kFloat32).min(0, keepDim = true).get(0) == [[0.0f, 2, -2, -4]].toRawTensor()
      check: b.to(kFloat32).min(1, keepDim = true).get(0) == [[-4.0f], [-2.0f]].toRawTensor()

    test "Max":
      let a = [2, -1, 3, -3, 5, 0].toRawTensor()
      check: a.max().item(int64) == 5
      check: a.to(kFloat32).max().item(float32) == 5.0f

      let b = [[1, 2, 3, -4], [0, 4, -2, 5]].toRawTensor()
      check: b.max(0, keepDim = true).get(0) == [[1, 4, 3, 5]].toRawTensor()
      check: b.max(1, keepDim = true).get(0) == [[3], [5]].toRawTensor()
      check: b.to(kFloat32).max(0, keepDim = true).get(0) == [[1.0f, 4, 3, 5]].toRawTensor()
      check: b.to(kFloat32).max(1, keepDim = true).get(0) == [[3.0f], [5.0f]].toRawTensor()

    test "Variance":
      let a = [-3.0, -2, -1, 0, 1, 2, 3].toRawTensor()
      check: abs(a.variance().item(float64) - 4.6666666666667) < 1e-8
      let b = [[1.0, 2, 3, -4], [0.0, 4, -2, 5]].toRawTensor()
      check: b.variance(0, keepDim = true) == [[0.5, 2.0, 12.5, 40.5]].toRawTensor()
      check: (
        b.variance(1, keepDim = true) -
        [[9.666666666666666], [10.91666666666667]].toRawTensor()
      ).abs().sum().item(float64) < 1e-8
    test "Standard Deviation":
      let a = [-3.0, -2, -1, 0, 1, 2, 3].toRawTensor()
      check: abs(a.stddev().item(float64) - 2.1602468994693) < 1e-8
      let b = [[1.0, 2, 3, -4], [0.0, 4, -2, 5]].toRawTensor()
      check: abs(
        b.stddev(0, keepDim = true) -
        [[0.7071067811865476, 1.414213562373095,
          3.535533905932738, 6.363961030678928]].toRawTensor()
      ).abs().sum().item(float64) < 1e-8
      check: abs(
        b.stddev(1, keepDim = true) -
        [[3.109126351029605], [3.304037933599835]].toRawTensor()
      ).abs().sum().item(float64) < 1e-8

    test "Argmax":
      let a = [[0, 4, 7],
                [1, 9, 5],
                [3, 4, 1]].toRawTensor
      check: argmax(a, 0, keepDim = true) == [[2, 1, 0]].toRawTensor
      check: argmax(a, 1, keepDim = true) == [[2],
                              [1],
                              [1]].toRawTensor

      block:
        let a = [[0, 1, 2],
                  [3, 4, 5]].toRawTensor
        check: argmax(a, 0, keepDim = true) == [[1, 1, 1]].toRawTensor
        check: argmax(a, 1, keepDim = true) == [[2],
                                [2]].toRawTensor

    test "Argmax_3D":
      let a = [
        [[1, 10, 5, 5, 7, 3], [8, 3, 7, 9, 3, 8], [5, 3, 7, 1, 4, 5], [8, 10, 5, 8, 9, 1], [10, 5, 2, 1, 5, 8]],
        [[10, 0, 1, 9, 0, 4], [5, 7, 10, 0, 7, 5], [6, 1, 1, 10, 2, 2], [6, 10, 1, 9, 7, 8], [10, 7, 5, 9, 1, 3]],
        [[9, 1, 2, 1, 5, 10], [6, 1, 7, 9, 3, 0], [2, 1, 4, 8, 5, 7], [5, 7, 0, 4, 3, 2], [2, 7, 5, 8, 5, 6]],
        [[2, 8, 5, 9, 1, 5], [5, 10, 6, 8, 0, 1], [0, 10, 0, 8, 6, 7], [5, 1, 4, 9, 3, 0], [1, 1, 4, 3, 9, 4]]
      ].toRawTensor

      check: argmax(a, 0, keepDim = true) == [
        [[1, 0, 0, 1, 0, 2], [0, 3, 1, 0, 1, 0], [1, 3, 0, 1, 3, 2], [0, 0, 0, 1, 0, 1], [0, 1, 1, 1, 3, 0]]
      ].toRawTensor

      check: argmax(a, 1, keepDim = true) == [
        [[4, 0, 1, 1, 3, 1]], [[0, 3, 1, 2, 1, 3]], [[0, 3, 1, 1, 0, 0]], [[1, 1, 1, 0, 4, 2]]
      ].toRawTensor

      check: argmax(a, 2, keepDim = true) == [
        [[1], [3], [2], [1], [0]], [[0], [2], [3], [1], [0]], [[5], [3], [3], [1], [3]], [[3], [1], [1], [3], [4]]
      ].toRawTensor

    block:
      let a = [
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
        [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]
      ].toRawTensor

      check: argmax(a, 0, keepDim = true) == [
        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
      ].toRawTensor

      check: argmax(a, 1, keepDim = true) == [
        [[2, 2, 2, 2]], [[2, 2, 2, 2]]
      ].toRawTensor

      check: argmax(a, 2, keepDim = true) == [
        [[3], [3], [3]], [[3], [3], [3]]
      ].toRawTensor

main()
GC_fullCollect()
