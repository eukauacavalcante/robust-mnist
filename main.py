import os
from datetime import datetime

import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.dataset import load_binary_mnist
from src.evaluation import (plot_confusion_matrix, plot_training_curves,
                            print_classification_report, save_metrics)
from src.model import build_model, compile_model, train_model
from src.preprocessing import preprocess

now = datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs("outputs/tensorboard", exist_ok=True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tb_callback = tf.keras.callbacks.TensorBoard(
    log_dir=f"outputs/tensorboard/run_{now}",
    histogram_freq=1,
    write_graph=True,
)

DATA_DIR = "data/"
OUTPUT_DIR = "outputs"

TRAIN_IMAGES = os.path.join(DATA_DIR, "train-images-idx3-ubyte")
TRAIN_LABELS = os.path.join(DATA_DIR, "train-labels-idx1-ubyte")
TEST_IMAGES = os.path.join(DATA_DIR, "t10k-images-idx3-ubyte")
TEST_LABELS = os.path.join(DATA_DIR, "t10k-labels-idx1-ubyte")

HIDDEN_UNITS = (512, 256, 128)
DROPOUT_RATE = 0.3
L2_LAMBDA = 1e-4
LEARNING_RATE = 1e-3
EPOCHS = 50
BATCH_SIZE = 256
VAL_SPLIT = 0.10
RANDOM_SEED = 42


print("\n" + "═" * 60)
print("ETAPA 1 - Carregamento dos Dados Brutos")
print("═" * 60)

x_train_raw, y_train = load_binary_mnist(TRAIN_IMAGES, TRAIN_LABELS)
x_test_raw, y_test = load_binary_mnist(TEST_IMAGES, TEST_LABELS)

print(f"  x_train_raw : shape={x_train_raw.shape}  dtype={x_train_raw.dtype}")
print(f"  y_train     : shape={y_train.shape}       dtype={y_train.dtype}")
print(f"  x_test_raw  : shape={x_test_raw.shape}   dtype={x_test_raw.dtype}")
print(f"  y_test      : shape={y_test.shape}        dtype={y_test.dtype}")

print("\n" + "═" * 60)
print("ETAPA 2 - Pré-processamento  (norm -> flatten)")
print("═" * 60)

x_train_full = preprocess(x_train_raw)
x_test = preprocess(x_test_raw)

print(f"  x_train_full: shape={x_train_full.shape}  dtype={x_train_full.dtype}")
print(f"  x_test      : shape={x_test.shape}        dtype={x_test.dtype}")
print(f"  Pixel range — min={x_train_full.min():.3f}  max={x_train_full.max():.3f}")

print("\n" + "═" * 60)
print("ETAPA 3 - Divisão Treino / Validação  (90% / 10%)")
print("═" * 60)

x_train, x_val, y_train_s, y_val = train_test_split(
    x_train_full,
    y_train,
    test_size=VAL_SPLIT,
    random_state=RANDOM_SEED,
    stratify=y_train,
)

print(f"  Treino    : {x_train.shape[0]} amostras  -> x_train: {x_train.shape}")
print(f"  Validação : {x_val.shape[0]} amostras  -> x_val  : {x_val.shape}")
print(f"  Teste     : {x_test.shape[0]} amostras  -> x_test : {x_test.shape}  [ISOLADO]")

print("\n" + "═" * 60)
print("ETAPA 4 - Arquitetura  RobustMNIST")
print("═" * 60)

model = build_model(
    input_dim=784,
    hidden_units=HIDDEN_UNITS,
    dropout_rate=DROPOUT_RATE,
    l2_lambda=L2_LAMBDA,
)
model = compile_model(model, learning_rate=LEARNING_RATE)
model.summary()

print("\n" + "═" * 60)
print("ETAPA 5 - Treinamento")
print("═" * 60)
print(f"  Épocas máx.: {EPOCHS}  |  Batch: {BATCH_SIZE}  |  Passos/época: {54000 // BATCH_SIZE + 1}")

start_time = datetime.now()

history = train_model(
    model=model,
    x_train=x_train,
    y_train=y_train_s,
    x_val=x_val,
    y_val=y_val,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[tb_callback],
)

end_time = datetime.now()

plot_training_curves(history, save_dir=os.path.join(OUTPUT_DIR, f"plots/{now}"))

print("\n" + "═" * 60)
print("ETAPA 6 - Avaliação no Teste Limpo  (dados inéditos)")
print("═" * 60)

loss_clean, acc_clean = model.evaluate(x_test, y_test, verbose=0)

print(f"  Loss    : {loss_clean:.4f}")
print(f"  Acurácia: {acc_clean * 100:.2f}%  (esperado: 97-99%)")

print_classification_report(model, x_test, y_test)
plot_confusion_matrix(model, x_test, y_test, label="limpo", save_dir=os.path.join(OUTPUT_DIR, f"plots{now}"))

metrics = {
    "dataset": "MNIST",
    "architecture": {"hidden_units": list(HIDDEN_UNITS), "dropout": DROPOUT_RATE, "l2": L2_LAMBDA},
    "epochs_run": len(history.history["loss"]),
    "final_val_loss": round(float(history.history["val_loss"][-1]), 5),
    "final_val_acc": round(float(history.history["val_accuracy"][-1]), 5),
    "test_loss": round(float(loss_clean), 5),
    "test_accuracy": round(float(acc_clean), 5),
}

save_metrics(metrics, save_dir=os.path.join(OUTPUT_DIR, f"metrics/{now}"))

train_time = end_time - start_time
train_time_formatted = str(train_time).split(".")[0] 

print("\n" + "═" * 60)
print("Pipeline concluído.")
print(f"  Acurácia final no teste: {acc_clean * 100:.2f}%")
print(f"  Tempo total de treinamento: {train_time_formatted}")
print("═" * 60)
