import struct

import numpy as np


def load_binary_mnist(image_paths, label_paths):
    with open(label_paths, 'rb') as lbpath:
        magic, num = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(image_paths, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8)
        images = images.reshape(len(labels), rows, cols)

    return images, labels
