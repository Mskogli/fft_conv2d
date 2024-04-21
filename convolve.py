import numpy as np
import matplotlib.pyplot as plt

from fft import fft2d, ifft2d, fftshift, ifftshift


def _pad_kernel(kernel: np.ndarray, image_shape: np.ndarray) -> np.ndarray:
    padded_kernel = np.zeros(image_shape)
    kernel_shape = kernel.shape
    padded_kernel[: kernel_shape[0], : kernel_shape[1]] = kernel
    return padded_kernel


def filter_image(
    image: np.ndarray,
    kernel: np.ndarray,
    skip_zero_freq_centering: bool = False,
    viz: bool = False,
) -> np.ndarray:
    padded_kernel = _pad_kernel(kernel, image.shape)

    # Compute the frequency domain representations and optionally shift the zero frequency to the origin
    freq_shift = fftshift if not skip_zero_freq_centering else lambda x: x
    inv_freq_shift = ifftshift if not skip_zero_freq_centering else lambda x: x

    kernel_freq_domain = freq_shift(fft2d(padded_kernel))
    image_freq_domain = freq_shift(fft2d(image))

    filtered_image = ifft2d(
        inv_freq_shift(image_freq_domain * kernel_freq_domain)
    ).real.astype(np.float32)

    if viz:
        _viz(image_freq_domain, kernel_freq_domain, filtered_image, image)

    return filtered_image


def _viz(
    image_freq_domain: np.ndarray,
    kernel_freq_domain: np.ndarray,
    filtered_image: np.ndarray,
    original_image: np.ndarray,
) -> None:
    plt.figure(figsize=(15, 7))
    ax1 = plt.subplot(1, 4, 1)
    ax1.imshow(np.log10(abs(image_freq_domain)).astype(np.float32), cmap="gray")
    ax1.set_title("Image Spectrum")
    ax1.axis("off")

    ax2 = plt.subplot(1, 4, 2)
    ax2.imshow(np.log10(abs(kernel_freq_domain)).astype(np.float32), cmap="gray")
    ax2.set_title("Kernel Spectrum")
    ax2.axis("off")

    ax3 = plt.subplot(1, 4, 3)
    ax3.imshow(filtered_image, cmap="gray")
    ax3.set_title("Filtered Image")
    ax3.axis("off")

    ax4 = plt.subplot(1, 4, 4)
    ax4.imshow(original_image, cmap="gray")
    ax4.set_title("Original Image")
    ax4.axis("off")

    plt.show()
