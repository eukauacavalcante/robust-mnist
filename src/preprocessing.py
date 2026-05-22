import numpy as np


def normalize(images: np.ndarray) -> np.ndarray:
    return images.astype(np.float32) / 255.0


def flatten(images: np.ndarray) -> np.ndarray:
    n_samples = images.shape[0]
    return images.reshape(n_samples, -1)


def preprocess(images: np.ndarray) -> np.ndarray:
    images = normalize(images)
    images = flatten(images)
    return images
