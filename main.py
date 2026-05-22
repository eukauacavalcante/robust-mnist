from src.dataset import load_binary_mnist
from src.preprocessing import preprocess

x_train, y_train = load_binary_mnist(
    image_paths="data/train-images-idx3-ubyte",
    label_paths="data/train-labels-idx1-ubyte",
)

x_train = preprocess(x_train)

# x_train.shape → (60000, 784)
# x_train.dtype → float32
# x_train.min()  → 0.0
# x_train.max()  → 1.0
