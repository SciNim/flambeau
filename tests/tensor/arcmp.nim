import flambeau

var a = @[
    @[1, 1, 1, 1, 1],
    @[2, 4, 8, 16, 32],
    @[3, 9, 27, 81, 243],
    @[4, 16, 64, 256, 1024],
    @[5, 25, 125, 625, 3125],
  ].toTensor

echo "0"
echo(a)

echo "1"
echo(a[2..^1|2, 3])

echo "2"
echo(a[_..^1|+1, 3])

echo "3"
a[0, _] = @[2, 2, 2, 2, 2].toRawTensor()
echo(a)

a[0, _] = @[3, 3, 3, 3, 3].toTensor()
echo(a)


