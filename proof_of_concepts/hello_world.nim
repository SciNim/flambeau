import ../flambeau

var t = eye(3)
t.print()
echo "\n-------------------------"

t += 10
t.print()
echo "\n-------------------------"

t.index(0, 1).print()
echo "\n-------------------------"
t.index(1, 1).print()
echo "\n-------------------------"


t.index_put(1, 1, 100)
t.print()
echo "\n-------------------------"

t.index_put(-1, -1, -100)
t.print()
echo "\n-------------------------"
