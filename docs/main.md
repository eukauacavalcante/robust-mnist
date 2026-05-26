# Explicação: `main.py`

O `main.py` é o orquestrador do pipeline. Ele não contém lógica própria: conecta os módulos `dataset.py`, `preprocessing.py` e `model.py` na ordem correta, passa os tensores entre eles e imprime o estado do sistema a cada etapa. A matemática de cada operação está documentada na documentação do módulo responsável.

---

## 1. Carregamento dos Dados Brutos

```python
x_train_raw, y_train = load_binary_mnist(
    image_paths="data/train-images-idx3-ubyte",
    label_paths="data/train-labels-idx1-ubyte",
)

x_test_raw, y_test = load_binary_mnist(
    image_paths="data/t10k-images-idx3-ubyte",
    label_paths="data/t10k-labels-idx1-ubyte",
)
```

`load_binary_mnist()` é chamada duas vezes: uma para o conjunto de treino e outra para o conjunto de teste. Cada chamada lê os arquivos binários `.ubyte` e devolve os tensores brutos, com pixels no intervalo $[0, 255]$ e `dtype=uint8`. O funcionamento interno da função está descrito em [dataset.md](/docs/dataset.md).

Os tensores retornados nesta etapa têm as seguintes dimensões:

| Variável | Shape | dtype |
|---|---|---|
| `x_train_raw` | (60000, 28, 28) | uint8 |
| `y_train` | (60000,) | uint8 |
| `x_test_raw` | (10000, 28, 28) | uint8 |
| `y_test` | (10000,) | uint8 |

---

## 2. Pré-processamento

```python
x_train_full = preprocess(x_train_raw)
x_test       = preprocess(x_test_raw)
```

`preprocess()` aplica normalização e flattening sobre cada tensor de imagens, conforme descrito em `preprocessing.md`. As dimensões resultantes são:

| Variável | Shape | dtype |
|---|---|---|
| `x_train_full` | (60000, 784) | float32 |
| `x_test` | (10000, 784) | float32 |

O conjunto de teste (`x_test`, `y_test`) é pré-processado aqui mas permanece isolado até a etapa de avaliação final. Ele não participa do treino nem da validação.

---

## 3. Divisão Treino e Validação

```python
x_train, x_val, y_train, y_val = train_test_split(
    x_train_full,
    y_train,
    test_size=0.10,
    random_state=42,
    stratify=y_train,
)
```

O conjunto de treino completo (60.000 amostras) é dividido em duas partições:

- **Treino** (`x_train`, `y_train`): 90% das amostras, 54.000 imagens. Usado para atualizar os pesos $W$ e o bias $b$ da rede a cada mini-batch.
- **Validação** (`x_val`, `y_val`): 10% das amostras, 6.000 imagens. Usado para monitorar a generalização ao final de cada época, sem influenciar os gradientes.

Dois parâmetros merecem atenção:

**`random_state=42`:** fixa a semente do gerador de números aleatórios, garantindo que a mesma divisão seja reproduzida em execuções diferentes.

**`stratify=y_train`:** preserva a proporção de cada dígito (0 a 9) nas duas partições. Sem estratificação, a divisão aleatória poderia gerar um conjunto de validação com distribuição de classes diferente do treino, distorcendo as métricas de avaliação.

---

## 4. Construção e Compilação do Modelo

```python
model = build_model(
    input_dim=784,
    hidden_units=(512, 256, 128),
    dropout_rate=0.3,
    l2_lambda=1e-4,
)
model = compile_model(model, learning_rate=1e-3)
```

`build_model()` instancia a arquitetura da rede e `compile_model()` configura o otimizador, a função de perda e a métrica de avaliação. A especificação matemática completa de ambas as funções está em [model.md](/docs/model.md)`.

Em seguida, `count_parameters()` contabiliza e imprime o total de parâmetros treináveis e não-treináveis. O volume esperado para esta configuração é de 571.018 parâmetros.

---

## 5. Treinamento

```python
history = train_model(
    model=model,
    x_train=x_train,
    y_train=y_train,
    x_val=x_val,
    y_val=y_val,
    epochs=50,
    batch_size=256,
)
```

`train_model()` executa o loop de otimização com mini-batches. Com 54.000 amostras e `batch_size=256`, cada época realiza $\lceil 54000 / 256 \rceil = 211$ passos de atualização dos pesos. O número máximo de épocas é 50, controlado por EarlyStopping e ReduceLROnPlateau, conforme descrito em [model.md](/docs/model.md).

O objeto `history` retornado registra `loss`, `accuracy`, `val_loss` e `val_accuracy` por época.

---

## 6. Avaliação no Conjunto de Teste Limpo

```python
loss_clean, acc_clean = model.evaluate(x_test, y_test, verbose=0)
```

O modelo é avaliado sobre `x_test` e `y_test`, que permaneceram isolados durante todas as etapas anteriores. Esta separação é o que confere validade estatística à métrica: o modelo nunca viu esses dados, seja para treinar pesos ou para guiar os callbacks. O resultado esperado para esta configuração é uma acurácia entre 97% e 99%.
