import keras
import numpy as np
from keras import layers, regularizers


def build_model(
    input_dim: int = 784,
    hidden_units: tuple[int, ...] = (512, 256, 128),
    dropout_rate: float = 0.3,
    l2_lambda: float = 1e-4,
) -> keras.Model:

    inputs = keras.Input(shape=(input_dim,), name="entrada_pixels")

    x = inputs

    for i, units in enumerate(hidden_units):
        x = layers.Dense(
            units=units,
            use_bias=True,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            kernel_regularizer=regularizers.l2(l2_lambda),
            name=f"dense_{i + 1}",
        )(x)
        x = layers.BatchNormalization(name=f"batchnorm_{i + 1}")(x)
        x = layers.Activation("relu", name=f"relu_{i + 1}")(x)
        x = layers.Dropout(rate=dropout_rate, name=f"dropout_{i + 1}")(x)

    outputs = layers.Dense(
        units=10,
        activation="softmax",
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        name="saida_softmax",
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="RobustMNIST")
    return model


def compile_model(model: keras.Model, learning_rate: float = 1e-3) -> keras.Model:

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_callbacks(patience: int = 5) -> list:

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    )

    return [early_stop, reduce_lr]


def train_model(
    model: keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 256,
) -> keras.callbacks.History:

    callbacks = get_callbacks(patience=5)

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )

    return history


def count_parameters(model: keras.Model) -> dict[str, int]:

    trainable = int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))
    non_trainable = int(np.sum([np.prod(v.shape) for v in model.non_trainable_weights]))
    return {
        "trainable": trainable,
        "non_trainable": non_trainable,
        "total": trainable + non_trainable,
    }
