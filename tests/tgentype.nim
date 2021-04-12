import flambeau
import flambeau/flambeau_raw

{.experimental: "views".} # TODO
proc main() =
  let tmp = [[-2, -6], [-12, -20]]
  echo tmp
  let tt = toTensor(tmp)
  echo tt

main()
