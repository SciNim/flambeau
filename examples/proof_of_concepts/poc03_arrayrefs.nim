import flambeau

var t = eye(5)
t.print()
echo "\n-------------------------"

echo t.sizes().asNimView()
echo t.strides().asNimView()
