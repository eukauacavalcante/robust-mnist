import os

from src.dataset import load_binary_mnist

directory = 'data'
x_train, y_train = load_binary_mnist(
    os.path.join(directory, 'train-images-idx3-ubyte'),
    os.path.join(directory, 'train-labels-idx1-ubyte')
)

print(f"Matriz de treino carregada! Formato original: {x_train.shape}")
print(f"Vetor de rótulos carregado! Formato: {y_train.shape}")
