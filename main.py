import os
import sys
import argparse
import json
import numpy as np
import imageio.v2 as imageio

from typing import Tuple
from convolve import filter_image
from kernels import Kernels, list_kernels


def _load_image(image_path: str) -> Tuple[np.ndarray, ...]:  # 2 tuple
    # Sanitize
    image_path = image_path.strip()  # rm leading and trailing ws
    image_path = os.path.normpath(image_path)  # rm double slashes

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"The provided image path does not exist: {image_path}")

    return imageio.imread(image_path, mode="F"), image_path


def _create_filtered_path(filepath: os.path) -> os.path:
    directory, filename = os.path.split(filepath)
    file_base, file_extension = os.path.splitext(filename)
    new_filename = f"{file_base}_filtered{file_extension}"
    new_filepath = os.path.join(directory, new_filename)
    return new_filepath


def _normalize_image(image: np.ndarray) -> np.ndarray:
    return ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).astype(
        np.uint8
    )


def _get_kernel(kernel: str, predfined_kernel: bool = False) -> np.ndarray:
    if not predfined_kernel:
        return np.array(json.loads(kernel))
    else:
        return Kernels[kernel]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter a greyscale image with a kernel using the convolution theorem"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        nargs="?",
        help="Specifiy the path to the image to be filtered",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        nargs="?",
        help="Specifiy the kernel to filter the image with as an ndarray formatted as a string",
    )
    parser.add_argument(
        "--predefined_kernel",
        action="store_true",
        help="Specifiy if the input provided to --kernel is a string indexing a predefined kernel, or a stand alone kernel definition",
    )
    parser.add_argument(
        "--no_centering",
        action="store_true",
        help="Skip zero centering of the frequencies before applying fourier transforms",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Vizualise frequency spectra and filtering results",
    )
    parser.add_argument(
        "--kernels", action="store_true", help="List available predifined kernels"
    )

    args = parser.parse_args()

    if args.kernels:
        list_kernels()
        sys.exit(0)

    if (args.image_path and args.kernel) is not None:
        kernel = _get_kernel(args.kernel, args.predefined_kernel)
        image, image_path = _load_image(args.image_path)

        if args.kernel == "SOBEL" and args.predefined_kernel:

            print("Filtering image with sobel operator")

            filtered_image_gx = filter_image(
                image,
                kernel["Gx"],
                skip_zero_freq_centering=args.no_centering,
                viz=args.viz,
            )
            filtered_image_gy = filter_image(
                image,
                kernel["Gy"],
                skip_zero_freq_centering=args.no_centering,
                viz=args.viz,
            )

            imageio.imsave(
                _create_filtered_path(image_path),
                _normalize_image(np.sqrt(filtered_image_gx**2 + filtered_image_gy**2)),
            )
        else:
            print("Filtering image with kernel: \n", kernel)

            filtered_image = filter_image(
                image,
                kernel,
                skip_zero_freq_centering=args.no_centering,
                viz=args.viz,
            )
            imageio.imsave(
                _create_filtered_path(image_path),
                _normalize_image(filtered_image),
            )

    else:
        raise Exception(
            "Please provide a path to an image and a kernel to filter the image with"
        )
