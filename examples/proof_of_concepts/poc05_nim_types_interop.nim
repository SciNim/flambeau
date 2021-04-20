import flambeau/flambeau_raw

{.experimental: "views".}
let a = [1, 2, 3, 4, 5]
let ta = a.toRawTensor()

ta.print()
echo "\n---------------"
