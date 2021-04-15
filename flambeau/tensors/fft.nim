import ../tensors
import ../raw/bindings/[rawtensors, c10]
import ../raw/cpp/[std_cpp]
import ../raw/sugar/[interop, indexing]
import std/[complex, macros]

type
  SomeComplex = Complex32|Complex64

{.push inline, noinit.}

# FFTSHIFT
# -----------------------------------------------------------------------
func fftshift*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.fftshift(convertRawTensor(self))
  )

func fftshift*[T](self: Tensor[T], dim: openArray[int64]): Tensor[T] =
  let dims = dim.asTorchView()
  convertTensor[T](
    rawtensors.ifftshift(convertRawTensor(self), dims)
  )

func ifftshift*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.ifftshift(convertRawTensor(self))
  )

func ifftshift*[T](self: Tensor[T], dim: openArray[int64]): Tensor[T] =
  let dims = dim.asTorchView()
  convertTensor[T](
    rawtensors.ifftshift(convertRawTensor(self), dims)
  )

let defaultNorm: CppString = initCppString("backward")

# TORCH FFT HEADER
# -----------------------------------------------------------------------
# Forward FFT :
#   float -> Complex[float]
#   Complex[float] -> Complex[float]
#
# Backward FFT :
#   float -> float
#   Complex[float] -> Complex[float]
#
# RFFT Can only do:
#   float -> Complex[float]
#
# FFT
# -----------------------------------------------------------------------
func fft*[T: SomeFloat](self: Tensor[T], n: int64, dim: int64 = -1, norm: CppString = defaultNorm): Tensor[Complex[T]] =
  ## Compute the 1-D Fourier transform
  ## ``n`` represent Signal length. If given, the input will either be zero-padded or trimmed to this length before computing the FFT.
  ## ``norm`` can be :
  ##    *[T] "forward" - normalize by 1/n
  ##    *[T] "backward" - no normalization
  ##    *[T] "ortho" - normalize by 1/sqrt(n)
  convertTensor[Complex[T]](
    rawtensors.fft(convertRawTensor(self), n, dim, norm)
  )

func fft*[T: SomeFloat](self: Tensor[T]): Tensor[Complex[T]] =
  ## Compute the 1-D Fourier transform
  convertTensor[Complex[T]](
    rawtensors.fft(convertRawTensor(self))
  )

func fft*[T: SomeComplex](self: Tensor[T], n: int64, dim: int64 = -1, norm: CppString = defaultNorm): Tensor[T] =
  ## Compute the 1-D Fourier transform
  ## ``n`` represent Signal length. If given, the input will either be zero-padded or trimmed to this length before computing the FFT.
  ## ``norm`` can be :
  ##    *[T] "forward" - normalize by 1/n
  ##    *[T] "backward" - no normalization
  ##    *[T] "ortho" - normalize by 1/sqrt(n)
  convertTensor[T](
    rawtensors.fft(convertRawTensor(self), n, dim, norm)
  )

func fft*[T: SomeComplex](self: Tensor[T]): Tensor[T] =
  ## Compute the 1-D Fourier transform
  convertTensor[T](
    rawtensors.fft(convertRawTensor(self))
  )

# IFFT
# -----------------------------------------------------------------------
func ifft*[T](self: Tensor[T], n: int64, dim: int64 = -1, norm: CppString = defaultNorm): Tensor[T] =
  ## Compute the 1-D Fourier transform
  ## ``n`` represent Signal length. If given, the input will either be zero-padded or trimmed to this length before computing the FFT.
  ## ``norm`` can be :
  ##   *[T] "forward" - no normalization
  ##   *[T] "backward" - normalization by 1/n
  ##   *[T] "ortho" - normalization by 1/sqrt(n)
  convertTensor[T](
    rawtensors.ifft(convertRawTensor(self), n, dim, norm)
  )

func ifft*[T](self: Tensor[T]): Tensor[T] =
  ## Compute the 1-D Fourier transform
  convertTensor[T](
    rawtensors.ifft(convertRawTensor(self))
  )

# FFT2
# -----------------------------------------------------------------------
func fft2*[T: SomeFloat](self: Tensor[T], s: openArray[int64], dim: openArray[int64], norm: CppString = defaultNorm): Tensor[Complex[T]] =
  ## Compute the 2-D Fourier transform
  ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
  ## ``norm`` can be :
  ##    *[Complex[T]] "forward" - normalize by 1/n
  ##    *[Complex[T]] "backward" - no normalization
  ##    *[Complex[T]] "ortho" - normalize by 1/sqrt(n)
  ## With n the logical FFT size: ``n = prod(s)``.
  let s = s.asTorchView()
  let dims = dim.asTorchView()
  convertTensor[Complex[T]](
    rawtensors.fft2(convertRawTensor(self), s, dims, norm)
  )

func fft2*[T: SomeFloat](self: Tensor[T], s: openArray[int64]): Tensor[Complex[T]] =
  ## Compute the 2-D Fourier transform
  ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
  let s = s.asTorchView()
  convertTensor[Complex[T]](
    rawtensors.fft2(convertRawTensor(self), s)
  )

func fft2*[T: SomeFloat](self: Tensor[T]): Tensor[Complex[T]] =
  ## Compute the 2-D Fourier transform
  convertTensor[Complex[T]](
    rawtensors.fft2(convertRawTensor(self))
  )

func fft2*[T: SomeComplex](self: Tensor[T], s: openArray[int64], dim: openArray[int64], norm: CppString = defaultNorm): Tensor[T] =
  ## Compute the 2-D Fourier transform
  ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
  ## ``norm`` can be :
  ##    * "forward" - normalize by 1/n
  ##    * "backward" - no normalization
  ##    * "ortho" - normalize by 1/sqrt(n)
  ## With n the logical FFT size: ``n = prod(s)``.
  let s = s.asTorchView()
  let dims = dim.asTorchView()
  convertTensor[T](
    rawtensors.fft2(convertRawTensor(self), s, dims, norm)
  )

func fft2*[T: SomeComplex](self: Tensor[T], s: openArray[int64]): Tensor[T] =
  ## Compute the 2-D Fourier transform
  ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
  let s = s.asTorchView()
  convertTensor[T](
    rawtensors.fft2(convertRawTensor(self), s)
  )

func fft2*[T: SomeComplex](self: Tensor[T]): Tensor[T] =
  ## Compute the 2-D Fourier transform
  convertTensor[T](
    rawtensors.fft2(convertRawTensor(self))
  )

# IFFT2
# -----------------------------------------------------------------------
func ifft2*[T](self: Tensor[T], s: openArray[int64], dim: openArray[int64], norm: CppString = defaultNorm): Tensor[T] =
  ## Compute the 2-D Inverse Fourier transform
  ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
  ## ``norm`` can be :
  ##   *[T: SomeComplex] "forward" - no normalization
  ##   *[T: SomeComplex] "backward" - normalization by 1/n
  ##   *[T: SomeComplex] "ortho" - normalization by 1/sqrt(n)
  ## With n the logical FFT size: ``n = prod(s)``.
  let s = s.asTorchView()
  let dims = dim.asTorchView()
  convertTensor[T](
    rawtensors.ifft2(convertRawTensor(self), s, dims, norm)
  )

func ifft2*[T](self: Tensor[T], s: openArray[int64]): Tensor[T] =
  ## Compute the 2-D Inverse Fourier transform
  ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
  let s = s.asTorchView()
  convertTensor[T](
    rawtensors.ifft2(convertRawTensor(self), s)
  )

func ifft2*[T](self: Tensor[T]): Tensor[T] =
  ## Compute the 2-D Inverse Fourier transform
  convertTensor[T](
    rawtensors.ifft2(convertRawTensor(self))
  )

# FFTN
# -----------------------------------------------------------------------
func fftn*[T: SomeFloat](self: Tensor[T], s: openArray[int64], dim: openArray[int64], norm: CppString = defaultNorm): Tensor[Complex[T]] =
  ## Compute the N-D Fourier transform
  ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
  ## ``norm`` can be :
  ##    *[Complex[T]] "forward" normalize by 1/n
  ##    *[Complex[T]] "backward" - no normalization
  ##    *[Complex[T]] "ortho" normalize by 1/sqrt(n)
  ## With n the logical FFT size: ``n = prod(s)``.
  let s = s.asTorchView()
  let dims = dim.asTorchView()
  convertTensor[Complex[T]](
    rawtensors.fftn(convertRawTensor(self), s, dims)
  )

func fftn*[T: SomeFloat](self: Tensor[T], s: openArray[int64]): Tensor[Complex[T]] =
  ## Compute the N-D Fourier transform
  ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
  let s = s.asTorchView()
  convertTensor[Complex[T]](
    rawtensors.fftn(convertRawTensor(self), s)
  )

func fftn*[T: SomeFloat](self: Tensor[T]): Tensor[Complex[T]] =
  ## Compute the N-D Fourier transform
  convertTensor[Complex[T]](
    rawtensors.fftn(convertRawTensor(self))
  )

func fftn*[T: SomeComplex](self: Tensor[T], s: openArray[int64], dim: openArray[int64], norm: CppString = defaultNorm): Tensor[T] =
  ## Compute the N-D Fourier transform
  ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
  ## ``norm`` can be :
  ##    *[T: SomeComplex] "forward" normalize by 1/n
  ##    *[T: SomeComplex] "backward" - no normalization
  ##    *[T: SomeComplex] "ortho" normalize by 1/sqrt(n)
  ## With n the logical FFT size: ``n = prod(s)``.
  let s = s.asTorchView()
  let dims = dim.asTorchView()
  convertTensor[T](
    rawtensors.fftn(convertRawTensor(self), s, dims)
  )

func fftn*[T: SomeComplex](self: Tensor[T], s: openArray[int64]): Tensor[T] =
  ## Compute the N-D Fourier transform
  ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
  let s = s.asTorchView()
  convertTensor[T](
    rawtensors.fftn(convertRawTensor(self), s)
  )

func fftn*[T: SomeComplex](self: Tensor[T]): Tensor[T] =
  ## Compute the N-D Fourier transform
  convertTensor[T](
    rawtensors.fftn(convertRawTensor(self))
  )

# IFFTN
# -----------------------------------------------------------------------
func ifftn*[T](self: Tensor[T], s: openArray[int64], dim: openArray[int64], norm: CppString = defaultNorm): Tensor[T] =
  ## Compute the N-D Inverse Fourier transform
  ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
  ## ``norm`` can be :
  ##   *[T: SomeComplex] "forward" - no normalization
  ##   *[T: SomeComplex] "backward" - normalization by 1/n
  ##   *[T: SomeComplex] "ortho" - normalization by 1/sqrt(n)
  ## With n the logical FFT size: ``n = prod(s)``.
  let s = s.asTorchView()
  let dims = dim.asTorchView()
  convertTensor[T](
    rawtensors.fftn(convertRawTensor(self), s, dims)
  )

func ifftn*[T](self: Tensor[T], s: openArray[int64]): Tensor[T] =
  ## Compute the N-D Inverse Fourier transform
  ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
  let s = s.asTorchView()
  convertTensor[T](
    rawtensors.fftn(convertRawTensor(self), s)
  )

func ifftn*[T](self: Tensor[T]): Tensor[T] =
  ## Compute the N-D Inverse Fourier transform
  convertTensor[T](
    rawtensors.ifftn(convertRawTensor(self))
  )

# RFFT
# -----------------------------------------------------------------------
func rfft*[T: SomeFloat](self: Tensor[T], n: int64, dim: int64 = -1, norm: CppString = defaultNorm): Tensor[Complex[T]] =
  ## Compute the 1-D Fourier transform of real-valued input
  ## ``n`` represent Signal length. If given, the input will either be zero-padded or trimmed to this length before computing the rfft.
  ## ``norm`` can be :
  ##    *[Complex[T]] "forward" - normalize by 1/n
  ##    *[Complex[T]] "backward" - no normalization
  ##    *[Complex[T]] "ortho" - normalize by 1/sqrt(n)
  convertTensor[Complex[T]](
    rawtensors.rfft(convertRawTensor(self), n, dim, norm)
  )

func rfft*[T: SomeFloat](self: Tensor[T]): Tensor[Complex[T]] =
  ## Compute the 1-D Fourier transform of real-valued input
  convertTensor[Complex[T]](
    rawtensors.rfft(convertRawTensor(self))
  )


# IRFFT
# -----------------------------------------------------------------------
func irfft*[T](self: Tensor[Complex[T]], n: int64, dim: int64 = -1, norm: CppString = defaultNorm): Tensor[T] =
  ## Compute the 1-D Fourier transform of real-valued input
  ## ``n`` represent Signal length. If given, the input will either be zero-padded or trimmed to this length before computing the rfft.
  ## ``norm`` can be :
  ##   *[T] "forward" - no normalization
  ##   *[T] "backward" - normalization by 1/n
  ##   *[T] "ortho" - normalization by 1/sqrt(n)
  convertTensor[T](
    rawtensors.irfft(convertRawTensor(self), n, dim, norm)
  )

func irfft*[T](self: Tensor[Complex[T]]): Tensor[T] =
  ## Compute the 1-D Fourier transform of real-valued input
  convertTensor[T](
    rawtensors.irfft(convertRawTensor(self))
  )

# RFFT2
# -----------------------------------------------------------------------
func rfft2*[T: SomeFloat](self: Tensor[T], s: openArray[int64], dim: openArray[int64], norm: CppString = defaultNorm): Tensor[Complex[T]] =
  ## Compute the 2-D Fourier transform of real-valued input
  ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the rfft.
  ## ``norm`` can be :
  ##    *[T] "forward" - normalize by 1/n
  ##    *[T] "backward" - no normalization
  ##    *[T] "ortho" - normalize by 1/sqrt(n)
  ## With n the logical rfft size: ``n = prod(s)``.
  let s = s.asTorchView()
  let dims = dim.asTorchView()
  convertTensor[Complex[T]](
    rawtensors.rfft2(convertRawTensor(self), s, dims, norm)
  )

func rfft2*[T: SomeFloat](self: Tensor[T], s: openArray[int64]): Tensor[Complex[T]] =
  ## Compute the 2-D Fourier transform of real-valued input
  ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the rfft.
  let s = s.asTorchView()
  convertTensor[Complex[T]](
    rawtensors.rfft2(convertRawTensor(self), s)
  )

func rfft2*[T: SomeFloat](self: Tensor[T]): Tensor[Complex[T]] =
  ## Compute the 2-D Fourier transform of real-valued input
  convertTensor[Complex[T]](
    rawtensors.rfft2(convertRawTensor(self))
  )

# IRFFT2
# -----------------------------------------------------------------------
func irfft2*[T](self: Tensor[Complex[T]], s: openArray[int64], dim: openArray[int64], norm: CppString = defaultNorm): Tensor[T] =
  ## Compute the 2-D Inverse Fourier transform of real-valued input
  ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the rfft.
  ## ``norm`` can be :
  ##   *[T] "forward" - no normalization
  ##   *[T] "backward" - normalization by 1/n
  ##   *[T] "ortho" - normalization by 1/sqrt(n)
  ## With n the logical rfft size: ``n = prod(s)``.
  let s = s.asTorchView()
  let dims = dim.asTorchView()
  convertTensor[T](
    rawtensors.irfft2(convertRawTensor(self), s, dims, norm)
  )

func irfft2*[T](self: Tensor[Complex[T]], s: openArray[int64]): Tensor[T] =
  ## Compute the 2-D Inverse Fourier transform of real-valued input
  ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the rfft.
  let s = s.asTorchView()
  convertTensor[T](
    rawtensors.irfft2(convertRawTensor(self), s)
  )

func irfft2*[T](self: Tensor[Complex[T]]): Tensor[T] =
  ## Compute the 2-D Inverse Fourier transform of real-valued input
  convertTensor[T](
    rawtensors.irfft2(convertRawTensor(self))
  )

# RFFTN
# -----------------------------------------------------------------------
func rfftn*[T: SomeFloat](self: Tensor[T], s: openArray[int64], dim: openArray[int64], norm: CppString = defaultNorm): Tensor[Complex[T]] =
  ## Compute the N-D Fourier transform of real-valued input
  ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the rfft.
  ## ``norm`` can be :
  ##    *[T] "forward" normalize by 1/n
  ##    *[T] "backward" - no normalization
  ##    *[T] "ortho" normalize by 1/sqrt(n)
  ## With n the logical rfft size: ``n = prod(s)``.
  let s = s.asTorchView()
  let dims = dim.asTorchView()
  convertTensor[Complex[T]](
    rawtensors.rfftn(convertRawTensor(self), s, dims)
  )

func rfftn*[T: SomeFloat](self: Tensor[T], s: openArray[int64]): Tensor[Complex[T]] =
  ## Compute the N-D Fourier transform of real-valued input
  ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the rfft.
  let s = s.asTorchView()
  convertTensor[Complex[T]](
    rawtensors.rfftn(convertRawTensor(self), s)
  )

func rfftn*[T: SomeFloat](self: Tensor[T]): Tensor[Complex[T]] =
  ## Compute the N-D Fourier transform of real-valued input
  convertTensor[Complex[T]](
    rawtensors.rfftn(convertRawTensor(self))
  )

# RFFTN
# -----------------------------------------------------------------------
func irfftn*[T](self: Tensor[Complex[T]], s: openArray[int64], dim: openArray[int64], norm: CppString = defaultNorm): Tensor[T] =
  ## Compute the N-D Inverse Fourier transform of real-valued input
  ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the rfft.
  ## ``norm`` can be :
  ##   *[T] "forward" - no normalization
  ##   *[T] "backward" - normalization by 1/n
  ##   *[T] "ortho" - normalization by 1/sqrt(n)
  ## With n the logical rfft size: ``n = prod(s)``.
  let s = s.asTorchView()
  let dims = dim.asTorchView()
  convertTensor[T](
    rawtensors.rfftn(convertRawTensor(self), s, dims)
  )

func irfftn*[T](self: Tensor[Complex[T]], s: openArray[int64]): Tensor[T] =
  ## Compute the N-D Inverse Fourier transform of real-valued input
  ## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the rfft.
  let s = s.asTorchView()
  convertTensor[T](
    rawtensors.rfftn(convertRawTensor(self), s)
  )

func irfftn*[T](self: Tensor[Complex[T]]): Tensor[T] =
  ## Compute the N-D Inverse Fourier transform
  convertTensor[T](
    rawtensors.irfftn(convertRawTensor(self))
  )

# func hfft*[T](self: Tensor[T], n: int64, dim: int64 = -1, norm: CppString = defaultNorm): Tensor[T] =
#   ## Computes the 1 dimensional FFT of a onesided Hermitian signal.
#   convertTensor[T](
#     rawtensors.hfft(convertRawTensor(self), n, dim, norm)
#   )
#
# func hfft*[T](self: Tensor[T]): Tensor[T] =
#   ## Computes the 1 dimensional FFT of a onesided Hermitian signal.
#   convertTensor[T](
#     rawtensors.hfft(convertRawTensor(self))
#   )
#
# func ihfft*[T](self: Tensor[T], n: int64, dim: int64 = -1, norm: CppString = defaultNorm): Tensor[T] =
#   ## Computes the inverse FFT of a real-valued Fourier domain signal.
#   convertTensor[T](
#     rawtensors.ihfft(convertRawTensor(self), n, dim, norm)
#   )
#
# func ihfft*[T](self: Tensor[T]): Tensor[T] =
#   ## Computes the inverse FFT of a real-valued Fourier domain signal.
#   convertTensor[T](
#     rawtensors.ihfft(convertRawTensor(self))
#   )
#

