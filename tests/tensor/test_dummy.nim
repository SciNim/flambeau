import ../../flambeau
import unittest
{.experimental: "views".} # TODO

proc simpleTest() =
  test "add_dim":
    var tt : Tensor[int] = [[-2, -6], [-12, -20]].toTensor()
    check tt.shape() == [2'i64, 2]
    let tt2 = tt.reshape(@[2'i64, 2, 1])
    check tt2.shape() == [2'i64, 2, 1]

proc flipCatTest() =
  test "Flip, Cat":
    var inttens: Tensor[int] = [[1, 2, 3], [4, 5, 6]].toTensor()
    echo "inttest", inttens
    var fliptens = flip(inttens, [1'i64])
    echo "fliptens", fliptens
    var catfliptens = cat(fliptens, inttens)
    echo "catfliptens", catfliptens


proc main() =
  simpleTest()
  flipCatTest()

main()
