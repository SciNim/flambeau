import flambeau/flambeau_raw
import std/math

{.experimental: "views".} # TODO this is ignored

const
    x = @[1, 2, 3, 4, 5]
    y = @[1, 2, 3, 4, 5]

var
    vandermonde: seq[seq[int]]
    row: seq[int]

vandermonde = newSeq[seq[int]]()

for i, xx in x:
    row = newSeq[int]()
    vandermonde.add(row)
    for j, yy in y:
        vandermonde[i].add(xx^yy)

let foo = vandermonde.toRawTensor()

foo.print()
echo "\n---------------"

let nest3 = @[
        @[
          [1,2,3],
          [1,2,3]
        ],
        @[
          [3,2,1],
          [3,2,1]
        ],
        @[
          [4,4,5],
          [4,4,4]
        ],
        @[
          [6,6,6],
          [6,6,6]
        ]
      ]

let t3 = nest3.toRawTensor()
t3.print()
echo "\n---------------"
