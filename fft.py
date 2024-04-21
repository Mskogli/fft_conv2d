# Naive implementation of the Cooley Tukey FFT algorithm
# https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm

import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt


def _fft(x: np.ndarray) -> np.ndarray:
    N = len(x)
    assert N > 0 and (N & (N - 1) == 0)

    if N <= 1:
        return x

    even = _fft(x[0::2])  # Down to base case of recursion, return x
    odd = _fft(x[1::2])  # Down to base case of recursion, return x

    T = [np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)]

    return np.array(
        [even[n] + T[n] for n in range(N // 2)]
        + [even[n] - T[n] for n in range(N // 2)]
    )


def _ifft(x: np.ndarray) -> np.ndarray:
    N = len(x)
    assert N > 0 and (N & (N - 1) == 0)

    if N <= 1:
        return x

    x_conjugate = np.conjugate(x)  # Not needed for image data but meh

    result = _fft(x_conjugate)
    result = np.conjugate(result)
    result = result / N

    return result


def fft2d(image: np.ndarray) -> np.ndarray:
    fft_rows = np.array([_fft(row) for row in image])
    return np.array([_fft(col) for col in fft_rows.T]).T


def ifft2d(image: np.ndarray) -> np.ndarray:
    ifft_rows = np.array([_ifft(row) for row in image])
    return np.array([_ifft(col) for col in ifft_rows.T]).T


def fftshift(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        n = len(x)
        return np.roll(x, n // 2)
    else:
        n1, n2 = x.shape
        return np.roll(np.roll(x, n1 // 2, axis=0), n2 // 2, axis=1)


def ifftshift(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        n = len(x)
        return np.roll(x, -n // 2)
    else:
        n1, n2 = x.shape
        return np.roll(np.roll(x, -n1 // 2, axis=0), -n2 // 2, axis=1)


if __name__ == "__main__":
    # Sanity check on the 2D FFT/IFFT implementation
    image = imageio.imread("imgs/lena.png", mode="F")

    np_fft2 = np.fft.fft2(image)
    np_ifft2 = np.fft.ifft2(np_fft2)

    my_fft2 = fft2d(image)
    my_ifft2 = ifft2d(my_fft2)

    assert np.allclose(my_fft2, np_fft2)
    assert np.allclose(my_ifft2, np_ifft2)
