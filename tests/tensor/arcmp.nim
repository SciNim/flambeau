include flambeau
#include flambeau/workaround

proc main() = 
  let c = [[1, 2], [3, 4]].toTensor()
  echo c
  echo c.flip[:int]( @[0.int64])
  echo c[_, 1]

main()