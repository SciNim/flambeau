# Testing C++ tuples and QR decomposition

import flambeau/flambeau_nn

{.experimental: "views".} # TODO this is ignored

proc main() =
  let a = [[12.0, -51, 4], [6.0, 167, -68], [-4.0, 24, -41]].toRawTensor()

  let qr = qr(a)

  let q = qr.get(0)
  let r = qr.get(1)

  echo "Q"
  q.print()
  echo "\n------------"
  echo "R"
  r.print()
  echo "\n------------"

main()
