# Copyright 2020.

# Copyright 2017-2021 the Flambeau contributors
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
import std/[unittest, math]

{.experimental: "views".} # TODO

proc main() =
  suite "Testing indexing and slice syntax":
    const
      a = @[1, 2, 3, 4, 5]
      b = @[1, 2, 3, 4, 5]

    var
      vandermonde: seq[seq[int]]
      row: seq[int]

    vandermonde = newSeq[seq[int]]()

    for i, aa in a:
      row = newSeq[int]()
      vandermonde.add(row)
      for j, bb in b:
        vandermonde[i].add(aa^bb)

    let t_van = vandermonde.toRawTensor()
    # echo t_van

    # Tensor of shape 5x5 of type "int" on backend "Cpu"
    # |1      1       1       1       1|
    # |2      4       8       16      32|
    # |3      9       27      81      243|
    # |4      16      64      256     1024|
    # |5      25      125     625     3125|

    test "Basic indexing - foo[2, 3]":
      check: t_van[2, 3].item(int) == 81

    test "Basic indexing - foo[1+1, 2*2*1]":
      check: t_van[1+1, 2*2*1].item(int) == 243

    ## TODO: The following checks fails because return value
    ## is a Tensor not a scalar
    # test "Indexing from end - foo[^1, 3]":
    #     check: t_van[^1, 3] == 625
    # test "Indexing from end - foo[^(1*2), 3]":
    #     check: t_van[^1, 3] == 256

    test "Basic slicing - foo[1..2, 3]":
      let test = @[16, 81]
      check: t_van[1..2, 3] == test.toRawTensor()

    test "Basic slicing - foo[1+1..4, 3-2..2]":
      let test = @[@[9, 27], @[16, 64], @[25, 125]]
      check: t_van[1+1..4, 3-2..2] == test.toRawTensor()

    test "Span slices - foo[_, 3]":
      let test = @[1, 16, 81, 256, 625]
      check: t_van[_, 3] == test.toRawTensor()

    test "Span slices - foo[1.._, 3]":
      let test = @[@[16], @[81], @[256], @[625]]
      check: t_van[1.._, 3] == test.toRawTensor().squeeze()

      ## Check with extra operators
      check: t_van[0+1.._, 3] == test.toRawTensor().squeeze()

    test "Span slices - foo[_..3, 3]":
      let test = @[@[1], @[16], @[81], @[256]]
      check: t_van[_..3, 3] == test.toRawTensor().squeeze()

      ## Check with extra operators
      check: t_van[_..5-2, 3] == test.toRawTensor().squeeze()

    test "Span slices - foo[_.._, 3]":
      let test = @[@[1], @[16], @[81], @[256], @[625]]
      check: t_van[_.._, 3] == test.toRawTensor().squeeze()

    test "Stepping - foo[1..3|2, 3]":
      let test = @[@[16], @[256]]
      check: t_van[1..3|2, 3] == test.toRawTensor().squeeze()
      check: t_van[1..3|+2, 3] == test.toRawTensor().squeeze()
      check: t_van[1*(0+1)..2+1|(5-3), 3] == test.toRawTensor().squeeze()

    test "Span stepping - foo[_.._|2, 3]":
      let test = @[@[1], @[81], @[625]]
      check: t_van[_.._|2, 3] == test.toRawTensor().squeeze()

    test "Span stepping - foo[_.._|+2, 3]":
      let test = @[@[1], @[81], @[625]]
      check: t_van[_.._|+2, 3] == test.toRawTensor().squeeze()

    test "Span stepping - foo[1.._|1, 2..3]":
      let test = @[@[8, 16], @[27, 81], @[64, 256], @[125, 625]]
      check: t_van[1.._|1, 2..3] == test.toRawTensor()

    test "Span stepping - foo[_..<4|2, 3]":
      let test = @[@[1], @[81]]
      check: t_van[_..<4|2, 3] == test.toRawTensor().squeeze()

    test "Slicing until at n from the end - foo[0..^4, 3]":
      let test = @[@[1], @[16]]
      check: t_van[0..^3, 3] == test.toRawTensor().squeeze()
      ## Check with extra operators
      check: t_van[0..^2+1, 3] == test.toRawTensor().squeeze()

    test "Span Slicing until at n from the end - foo[_..^2, 3]":
      let test = @[@[1], @[16], @[81], @[256]]
      check: t_van[_..^1, 3] == test.toRawTensor().squeeze()
      ## Check with extra operators
      check: t_van[_..^1+0, 3] == test.toRawTensor().squeeze()

    test "Stepped Slicing until at n from the end - foo[1..^1|2, 3]":
      let test = @[@[16], @[256]]
      check: t_van[1..^1|2, 3] == test.toRawTensor().squeeze()
      ## Check with extra operators
      check: t_van[1..^1|(1+1), 3] == test.toRawTensor().squeeze()

    # ##############################################
    # Disabled : see README
    # Slice from the end not working
    # ##############################################
    #
    # test "Slice from the end - foo[^1..0|-1, 3]":
    #   let test = @[@[625],@[256],@[81],@[16],@[1]]
    #   check: t_van[^1..0|-1, 3] == test.toRawTensor().squeeze()
    #   ## Check with extra operators
    #   let test2 = @[@[256],@[81],@[16],@[1]]
    #   check: t_van[^(4-2)..0|-1, 3] == test2.toRawTensor().squeeze()
    # when compileOption("boundChecks") and not defined(openmp):
    #   test "Slice from the end - expect non-negative step error - foo[^1..0, 3]":
    #     expect(IndexDefect):
    #       discard t_van[^1..0, 3]
    # else:
    #   echo "Bound-checking is disabled or OpenMP is used. The Slice from end, non-negative step error test has been skipped."

    # test "Slice from the end - foo[^(2*2)..2*2, 3]":
    #   let test = @[@[16],@[81],@[256],@[625]]
    #   check: t_van[^(2*2)..2*2, 3] == test.toRawTensor().squeeze()

    # test "Slice from the end - foo[^3..^2, 3]":
    #   let test = @[@[81],@[256]]
    #   check: t_van[^3..^2, 3] == test.toRawTensor().squeeze()
    #
    # ##############################################
    # End disabled section
    # ##############################################

  suite "Slice mutations":
    const
      a = @[1, 2, 3, 4, 5]
      b = @[1, 2, 3, 4, 5]

    var
      vandermonde: seq[seq[int]]
      row: seq[int]

    vandermonde = newSeq[seq[int]]()

    for i, aa in a:
      row = newSeq[int]()
      vandermonde.add(row)
      for j, bb in b:
        vandermonde[i].add(aa^bb)

    let t_van_immut = vandermonde.toRawTensor().squeeze()

    # Tensor of shape 5x5 of type "int" on backend "Cpu"
    # |1      1       1       1       1|
    # |2      4       8       16      32|
    # |3      9       27      81      243|
    # |4      16      64      256     1024|
    # |5      25      125     625     3125|

    test "Immutability - let variables cannot be changed":
      when compiles(t_van_immut[1..2, 3..4] = 999):
        check false
      when compiles(t_van_immut[0..1, 0..1] = [111, 222, 333, 444]):
        check false
      when compiles(t_van_immut[0..1, 0..1] = t_van_immut[111, 222, 333, 444]):
        check false

    test "Setting a slice to a single value":
      var t_van = t_van_immut.clone
      let test = @[@[1, 1, 1, 1, 1],
                    @[2, 4, 8, 999, 999],
                    @[3, 9, 27, 999, 999],
                    @[4, 16, 64, 256, 1024],
                    @[5, 25, 125, 625, 3125]]

      let t_test = test.toRawTensor().squeeze()
      t_van[1..2, 3..4] = 999
      check: t_van == t_test

    test "Setting a slice to an array/seq of values":
      var t_van = t_van_immut.clone
      let test = @[@[111, 222, 1, 1, 1],
              @[333, 444, 8, 16, 32],
              @[3, 9, 27, 81, 243],
              @[4, 16, 64, 256, 1024],
              @[5, 25, 125, 625, 3125]]

      let t_test = test.toRawTensor().squeeze()
      t_van[0..1, 0..1] = [[111, 222], [333, 444]].toRawTensor
      check: t_van == t_test

    test "Setting a slice from a different Tensor":
      var t_van = t_van_immut.clone
      let test = @[
        @[1, 1, 1, 1, 1],
        @[2, 16, 64, 256, 32],
        @[3, 25, 125, 625, 243],
        @[4, 4, 8, 16, 1024],
        @[5, 9, 27, 81, 3125],
      ]

      let t_test = test.toRawTensor().squeeze()
      t_van[3..4, 1..3] = t_van_immut[1..2, 1..3|+1]
      t_van[1..2, 1..3] = t_van_immut[3..4, 1..3|1]
      check: t_van == t_test

    test "Slicing from the end":
      # Other slicing from the end are forbidden because we cannot prove at CT that it does not result in slicing with negative steps
      var t_van = t_van_immut.clone
      check: t_van[^2..^1, 2..4] == @[@[27, 81, 243], @[64, 256, 1024]].toRawTensor()

    when compileOption("boundChecks") and not defined(openmp) and false:
      # No bound checking implemented for now
      test "Bounds checking":
        var t_van = t_van_immut.clone
        expect(IndexDefect):
          t_van[0..1, 0..1] = [111, 222, 333, 444, 555].toRawTensor().squeeze()
        expect(IndexDefect):
          t_van[0..1, 0..1] = [111, 222, 333].toRawTensor().squeeze()
        expect(IndexDefect):
          t_van[^2..^1, 2..4] = t_van[1, 4..2|-1]
        expect(IndexDefect):
          t_van[^2..^1, 2..4] = t_van[^1..^3|-1, 4..2|-1]
    else:
      echo "Bound-checking is disabled or OpenMP is used. The Out of bound checking test has been skipped."

    test "Chained slicing - foo[1..^1,1..2][1.._, 0]":
      let t_van = t_van_immut
      check: t_van[1..^1, 1..2][1.._, 0] == [[9], [16]].toRawTensor().squeeze()

  # TODO : implement atAxisIndex
  # suite "Axis slicing":
  #   let a =  [[  1,  2,  3,  4,  5,  6],
  #             [  7,  8,  9, 10, 11, 12],
  #             [ 13, 14, 15, 16, 17, 18]].toRawTensor().squeeze()
  #   test "atAxisIndex slicing":

  #     check:
  #       a.atAxisIndex(0, 0) == [[  1,  2,  3,  4,  5,  6]].toRawTensor().squeeze()
  #       a.atAxisIndex(0, 1) == [[  7,  8,  9, 10, 11, 12]].toRawTensor().squeeze()
  #       a.atAxisIndex(0, 2) == [[ 13, 14, 15, 16, 17, 18]].toRawTensor().squeeze()

  #       a.atAxisIndex(0, 0, 2) ==  [[  1,  2,  3,  4,  5,  6],
  #                                   [  7,  8,  9, 10, 11, 12]].toRawTensor().squeeze()
  #       a.atAxisIndex(0, 1, 2) ==  [[  7,  8,  9, 10, 11, 12],
  #                                   [ 13, 14, 15, 16, 17, 18]].toRawTensor().squeeze()

  #       a.atAxisIndex(1, 0) == [[1],
  #                               [7],
  #                               [13]].toRawTensor().squeeze()
  #       a.atAxisIndex(1, 1, 2) ==  [[2, 3],
  #                                   [8, 9],
  #                                   [14, 15]].toRawTensor().squeeze()

  #   when compileOption("boundChecks") and not defined(openmp):
  #     test "atAxisIndex bounds checking":
  #       expect(IndexDefect):
  #         echo a.atAxisIndex(0, 3)
  #       expect(IndexDefect):
  #         echo a.atAxisIndex(1, 3, 6)

main()
