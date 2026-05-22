# Explicação: `src/preprocessing.py`

Este documento descreve como o script `preprocessing.py` trabalha com os tensores recebidos do `dataset.py`, realizando a normalização dos pixels e aplicando o *flattening*.

---

## Normalização (`normalize()`)

O tensor que chega do `dataset.py` tem dtype(`uint8`), ou seja, segue valores que alternam entre 0 e 255. Para alimentar a rede neural e garantir a precisão do modelo, é necessário transformar o tensor em `float32` e mapear esse intervalo para $[0, 1]$:

$$\mathcal{X_{norm}} = \mathcal{X} \div 255.0$$

**Divisão escalar**: A função percorre todos os 47.040.000 de elementos e os divide por $255.0$.

``` python
images.astype(np.float32) / 255.0
```

### Por que `float32`?

Se o tensor utilizado no mapeamento for `uint8`, será perdido os valores de cinza, pois o valor final sempre será um inteiro:

$$128_{uint8} \div 255 = 0_{uint8}$$
$$128.0_{float32} \div 255.0 = 0.502_{float32}$$

## Flatting (`flatten()`)

A camada do *Dense* do **Keras** opera com a seguinte multiplicação:

$$y = Wx + b, W \in \mathbb{R_{m \times n}}, x \in \mathbb{R_{n \times 1}}$$

Para isso, $x$ precisa ser um vetor unidimensional (1D). Na multiplicação de matrizes, o número de colunas da matriz $A$ precisa ser igual ao número de linhas da matriz $B$. A imagem normalizada ainda tem forma $28 \times 28$, o *Flatten* reinterpreta a forma do tensor, colapsando as duas últimas dimensões (28, 28) em uma só (784). Tensor final: (60000, 784).

$$\mathcal{X[i]} \in \mathbb{R_{28 \times 28}} \xrightarrow{flatten} x_i \in \mathbb{R_{784}}$$

A linha de código abaixo é responsável por fixar o valor da primeira dimensão na variável `n_samples`.

``` python
# dimensões: ([0]60000, [1]784)
# dimensão [0] (primeira): 60000

n_samples = images.shape[0] # n_samples = 60000
```
O NumPy sabe que o tensor tem 47.040.000 elementos no total e que a primeira dimensão foi fixada em 60.000. Então ele resolve a divisão:

``` python
return images.reshape(n_samples, -1)
```

$$47.040.000 \div 60.000 = 784$$

E substitui o -1 por 784 automaticamente. Isso é matematicamente equivalente a:

``` python
images.reshape(60000, 784)
```

Mas com uma vantagem: se amanhã o dataset tiver imagens de 32×32 em vez de 28×28, não precisa alterar o código. O -1 sempre vai calcular o valor correto sozinho.
