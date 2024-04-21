# Some common image processing kernels https://en.wikipedia.org/wiki/Kernel_(image_processing)

import numpy as np

Kernels = {
    "GAUSSIAN_BLUR_3x3": 1 / 16 * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]),
    "GAUSSIAN_BLUR_5x5": 1
    / 256
    * np.array(
        [
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1],
        ]
    ),
    "UNSHARP_MASKING": -1
    / 256
    * np.array(
        [
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, -476, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1],
        ]
    ),
    "SHARPEN": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
    "EDGE": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
    "BOX_BLUR": 1 / 9 * np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
    "SOBEL": {
        "Gx": np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]),
        "Gy": np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]),
    },
}


def list_kernels():
    for kernel in Kernels:
        print(kernel)
