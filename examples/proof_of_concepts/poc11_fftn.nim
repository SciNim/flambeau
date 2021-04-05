import flambeau

{.experimental: "views".}
proc flambeau_fft*(shape: openArray[int64]) =
  var input = rand(shape.asTorchView(), kfloat64)
  var output = fftn(input)

  echo output.sizes().asNimView()
  echo output

flambeau_fft([2.int64, 3.int64, 4.int64])
