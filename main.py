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


if __name__ == '__main__':
    with Image.open(sys.argv[1]) as im:
        im = preprocess_image(im)
        A = np.array(im)
        E = haar_encode(A)
        fig, axes = plt.subplots(1, 4)
        for tol, ax in zip(range(1, 9, 2), axes.reshape(-1)):
            E = truncate_values(E, tol)  # Reuse E across iterations because tolerance increases
            D = haar_decode(E)
            ax.imshow(D, cmap='gray')
            ax.set_title(f'({tol}) 1:{(A != 0).sum() / (E != 0).sum():.1f}')
            ax.tick_params(which='both', bottom=False, top=False, left=False, right=False,
                           labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        fig.tight_layout()
        plt.show()
        fig, ax = plt.subplots()
        E = truncate_values(E, 12)
        D = haar_decode(E)
        ax.imshow(D, cmap='gray')
        ax.set_title(f'(12) 1:{(A != 0).sum() / (E != 0).sum():.1f}')
        ax.tick_params(which='both', bottom=False, top=False, left=False, right=False,
                       labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        fig.tight_layout()
        plt.show()
