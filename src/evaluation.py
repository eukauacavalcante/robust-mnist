import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def plot_training_curves(history, save_dir: str = "outputs/plots") -> None:
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(history.history["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Curvas de Aprendizado - RobustMNIST", fontsize=14, fontweight="bold")

    axes[0].plot(
        epochs,
        history.history["loss"],
        label="Treino",
        color="#2563eb",
        linewidth=2
    )
    axes[0].plot(
        epochs,
        history.history["val_loss"],
        label="Validação",
        color="#dc2626",
        linewidth=2,
        linestyle="--"
    )
    axes[0].set_title("Perda por Época")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Perda")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        epochs,
        history.history["accuracy"],
        label="Treino",
        color="#2563eb",
        linewidth=2
    )
    axes[1].plot(
        epochs,
        history.history["val_accuracy"],
        label="Validação",
        color="#dc2626",
        linewidth=2,
        linestyle="--"
    )
    axes[1].set_title("Acurácia por Época")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Acurácia")
    axes[1].set_ylim([0.8, 1.01])
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    path = os.path.join(save_dir, "training_curves.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluation] Curvas salvas em: {path}")


def plot_confusion_matrix(
    model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    label: str = "limpo",
    save_dir: str = "outputs/plots",
) -> np.ndarray:
    os.makedirs(save_dir, exist_ok=True)

    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    cm = confusion_matrix(y_test, y_pred)

    # Normalização por linha: C_norm[i,j] = C[i,j] / Σ_j C[i,j]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Matriz de Confusão - Dataset {label.upper()}", fontsize=13, fontweight="bold")

    classes = [str(i) for i in range(10)]

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Contagem absoluta  C[i,j]", "Normalizada por linha  C_norm[i,j]"],
        ["d", ".2f"],
    ):
        im = ax.imshow(data, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Classe Prevista  (j)")
        ax.set_ylabel("Classe Real  (i)")
        ax.set_xticks(range(10))
        ax.set_xticklabels(classes)
        ax.set_yticks(range(10))
        ax.set_yticklabels(classes)

        thresh = data.max() / 2.0
        for i in range(10):
            for j in range(10):
                ax.text(
                    j, i, format(data[i, j], fmt),
                    ha="center", va="center", fontsize=7,
                    color="white" if data[i, j] > thresh else "black",
                )

    path = os.path.join(save_dir, f"confusion_matrix_{label}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluation] Matriz de confusão salva em: {path}")
    return cm


def print_classification_report(model, x_test: np.ndarray, y_test: np.ndarray) -> None:
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    print("\n" + "=" * 60)
    print("RELATÓRIO DE CLASSIFICAÇÃO POR DÍGITO")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)]))


def save_metrics(metrics: dict, save_dir: str = "outputs/metrics") -> None:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[evaluation] Métricas salvas em: {path}")
