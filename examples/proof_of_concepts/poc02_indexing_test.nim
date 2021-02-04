import ../flambeau

var t = eye(5)
t.print()
echo "\n-------------------------"

let t2 = t[1..2, 1..2]
t2.print()
echo "\n-------------------------"

let t3 = t[1..2, 1.._]
t3.print()
echo "\n-------------------------"

# Argh, PyTorch doesn't support negative slices
#
# let t4 = t[^2..0|-1, 3]
# t4.print()
# echo "\n-------------------------"
