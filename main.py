import math
import sys

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageOps


def preprocess_image(image: Image) -> Image:
    """
    Create a square matrix with side length that's a power of 2.
    """
    image = ImageOps.grayscale(image)  # convert to grayscale first so padding is less expensive
    dim = max(image.size)  # Find the largest dimension
    new_dim = 2 ** int(math.ceil(math.log(dim, 2)))  # Find the next power of 2
    return ImageOps.pad(image, (new_dim, new_dim))


def get_haar_step(i: int, k: int) -> np.ndarray:
    transform = np.zeros((2 ** k, 2 ** k))
    # Averages
    for j in range(2 ** (k - i - 1)):
        transform[2 * j, j] = 1 / 2
        transform[2 * j + 1, j] = 1 / 2
    # Details
    offset = 2 ** (k - i - 1)
    for j in range(2 ** (k - i - 1)):
        transform[2 * j, offset + j] = 1 / 2
        transform[2 * j + 1, offset + j] = -1 / 2
    # Identity
    for j in range(2 ** (k - i), 2 ** k):
        transform[j, j] = 1
    return transform


def get_haar_transform(k: int) -> np.ndarray:
    transform = np.eye(2 ** k)
    for i in range(k):
        transform = transform @ get_haar_step(i, k)
    return transform


def haar_encode(a: np.ndarray) -> np.ndarray:
    k = int(math.log2(len(a)))
    row_encoder = get_haar_transform(k)
    return row_encoder.T @ a @ row_encoder


def haar_decode(a: np.ndarray) -> np.ndarray:
    k = int(math.log2(len(a)))
    row_decoder = np.linalg.inv(get_haar_transform(k))
    return row_decoder.T @ a @ row_decoder


def truncate_values(a: np.ndarray, tolerance: float) -> np.ndarray:
    return np.where(np.abs(a) < tolerance, 0, a)


def calculate_compression_ratio(original, compressed) -> float:
    return (original != 0).sum() / (compressed != 0).sum()


def plot_image(original: np.array, encoded: np.array, tolerance: float, axes_subplot) -> None:
    encoded = truncate_values(encoded,
                              tolerance)  # Reuse E across iterations because tolerance increases
    decoded = haar_decode(encoded)
    axes_subplot.imshow(decoded, cmap='gray')
    axes_subplot.set_title(f'({tolerance}) 1:{calculate_compression_ratio(original, encoded) :.1f}')
    axes_subplot.tick_params(which='both', bottom=False, top=False, left=False, right=False,
                             labelbottom=False, labeltop=False, labelleft=False, labelright=False)


if __name__ == '__main__':
    with Image.open(sys.argv[1]) as im:
        im = preprocess_image(im)
        A = np.array(im)
        E = haar_encode(A)
        fig, axes = plt.subplots(1, 4)
        for tol, ax in zip(range(1, 9, 2), axes.reshape(-1)):
            plot_image(A, E, tol, ax)
        fig.tight_layout()
        plt.show()
        fig, ax = plt.subplots()
        plot_image(A, E, 12, ax)
        fig.tight_layout()
        plt.show()
