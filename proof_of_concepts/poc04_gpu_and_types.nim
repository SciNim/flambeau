import ../flambeau

let t_cuda = eye(5, kCuda)
t_cuda.print()
echo "\n-------------------------"

let t_int8 = eye(5, kInt8)
t_int8.print()
echo "\n-------------------------"
