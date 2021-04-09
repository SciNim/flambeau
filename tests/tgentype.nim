import flambeau

proc main() =
  var tt = toTensor([[-2, -6], [-12, -20]])
  echo tt.sizes()
  echo tt

main()
