from sklearn.model_selection import train_test_split

from src.dataset import load_binary_mnist
from src.model import build_model, compile_model, count_parameters, train_model
from src.preprocessing import preprocess

print(">>> [1/5] Carregando dados binários MNIST...")

x_train_raw, y_train = load_binary_mnist(
    image_paths="data/train-images-idx3-ubyte",
    label_paths="data/train-labels-idx1-ubyte",
)

x_test_raw, y_test = load_binary_mnist(
    image_paths="data/t10k-images-idx3-ubyte",
    label_paths="data/t10k-labels-idx1-ubyte",
)

print(f"    Treino bruto: {x_train_raw.shape} | dtype: {x_train_raw.dtype}")
print(f"    Teste  bruto: {x_test_raw.shape}  | dtype: {x_test_raw.dtype}")

print("\n>>> [2/5] Pré-processando tensores...")

x_train_full = preprocess(x_train_raw)  # (60000, 784), float32
x_test = preprocess(x_test_raw)  # (10000, 784), float32

print(f"    Treino processado: {x_train_full.shape} | dtype: {x_train_full.dtype}")
print(f"    Teste  processado: {x_test.shape}       | dtype: {x_test.dtype}")
print("\n>>> [3/5] Dividindo treino e validação (90% / 10%)...")

x_train, x_val, y_train, y_val = train_test_split(
    x_train_full,
    y_train,
    test_size=0.10,
    random_state=42,
    stratify=y_train,
)

print(f"    Treino   : {x_train.shape}")
print(f"    Validação: {x_val.shape}")
print("\n>>> [4/5] Construindo e compilando o modelo...")

model = build_model(
    input_dim=784,
    hidden_units=(512, 256, 128),
    dropout_rate=0.3,
    l2_lambda=1e-4,
)
model = compile_model(model, learning_rate=1e-3)

params = count_parameters(model)
print(f"    Parâmetros treináveis    : {params['trainable']:,}")
print(f"    Parâmetros não-treináveis: {params['non_trainable']:,}")
print(f"    Total                    : {params['total']:,}")
model.summary()

print("\n>>> [5/5] Treinando o modelo...")

history = train_model(
    model=model,
    x_train=x_train,
    y_train=y_train,
    x_val=x_val,
    y_val=y_val,
    epochs=50,
    batch_size=256,
)

print("\n>>> Avaliando no conjunto de teste limpo (MNIST)...")

loss_clean, acc_clean = model.evaluate(x_test, y_test, verbose=0)

print(f"    Loss     : {loss_clean:.4f}")
print(f"    Acurácia : {acc_clean * 100:.2f}%")
print("    Pipeline encerrado com sucesso.")
