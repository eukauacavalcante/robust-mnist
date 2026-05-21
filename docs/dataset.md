# Explicação: `src/dataset.py`

Este documento descreve matematicamente cada transformação de dados que ocorre no script, desde a leitura dos arquivos binários `.ubyte` até a obtenção dos tensores prontos para uso em uma rede neural.

---

## Contexto: o formato IDX

O dataset MNIST é distribuído em arquivos binários no formato IDX. Esses arquivos não são imagens convencionais. São sequências brutas de bytes organizadas com um cabeçalho inicial de metadados seguido de dados puros.

Existem dois tipos de arquivo utilizados no script:

| Arquivo | Cabeçalho | Conteúdo |
|---|---|---|
| `train-labels-idx1-ubyte` | 8 bytes (2 inteiros) | 60.000 bytes de rótulos |
| `train-images-idx3-ubyte` | 16 bytes (4 inteiros) | 47.040.000 bytes de pixels |

---

## Etapa 1: Leitura dos rótulos

```python
magic, n = struct.unpack('>II', lbpath.read(8))
labels = np.fromfile(lbpath, dtype=np.uint8)
```

### Decodificação do cabeçalho

A chamada `struct.unpack('>II', ...)` lê os primeiros 8 bytes do arquivo e os interpreta como dois inteiros de 4 bytes no formato Big-Endian (`>`):

$$\underbrace{b_0\ b_1\ b_2\ b_3}_{\texttt{magic}} \quad \underbrace{b_4\ b_5\ b_6\ b_7}_{n = 60000}$$

O valor `magic` é um número de controle do formato IDX e não é utilizado na sequência. O valor `n` informa quantos rótulos existem no arquivo, ou seja, 60.000.

### Leitura dos dados

Após o cabeçalho, existe uma sequência linear de 60.000 rótulos. Cada rótulo ocupa exatamente 1 byte sem sinal (`uint8`), representando um dígito de 0 a 9.

> **Byte sem sinal**: não existem números negativos nesse formato. O intervalo de valores é $[0, 255]$. No caso dos rótulos, apenas os valores $[0, 9]$ são válidos.

Isso significa que cada rótulo possui 1 byte que representa um número inteiro no intervalo $[0, 9]$, como $r_4 = 00000100$ para o dígito 4, ou $r_9 = 00001001$ para o dígito 9. Um dígito pode aparecer múltiplas vezes, então o arquivo pode conter bytes repetidos dentro de 60.000 posições.

O NumPy lê o restante do arquivo e produz um vetor unidimensional:

$$\mathbf{y}_{\text{train}} = \begin{bmatrix} 5 & 0 & 4 & 1 & 9 & \cdots & 8 \end{bmatrix}$$

$$\mathbf{y}_{\text{train}} \in \mathbb{Z}^{60000}, \quad y_i \in \{0, 1, 2, \ldots, 9\}$$

Verificação do formato:

```
y_train.shape → (60000,)
```

---

## Etapa 2: Leitura das imagens

```python
magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
images = np.fromfile(imgpath, dtype=np.uint8)
```

### Decodificação do cabeçalho

O arquivo de imagens possui um cabeçalho maior, de 16 bytes, contendo quatro inteiros:

$$\underbrace{b_0 \cdots b_3}_{\texttt{magic}} \quad \underbrace{b_4 \cdots b_7}_{\texttt{num} = 60000} \quad \underbrace{b_8 \cdots b_{11}}_{\texttt{rows} = 28} \quad \underbrace{b_{12} \cdots b_{15}}_{\texttt{cols} = 28}$$

**`magic`**: número de controle do formato IDX, não utilizado posteriormente.

**`num`**: número de imagens, 60.000.

**`rows`**: número de linhas por imagem, 28.

**`cols`**: número de colunas por imagem, 28.

### Leitura dos dados brutos

Após o cabeçalho, os pixels de todas as imagens estão concatenados sequencialmente. Cada pixel ocupa 1 byte (`uint8`), com valor inteiro no intervalo $[0,\ 255]$.

O total de bytes lidos é:

$$60000 \times 28 \times 28 = 47.040.000 \text{ bytes}$$

O NumPy os organiza em um vetor unidimensional gigante:

$$\mathbf{v} = \begin{bmatrix} P^{(0)}_{1,1} & P^{(0)}_{1,2} & \cdots & P^{(0)}_{28,28} & P^{(1)}_{1,1} & \cdots & P^{(59999)}_{28,28} \end{bmatrix}$$

$$\mathbf{v} \in \mathbb{Z}^{47.040.000}$$

Neste estado, o vetor não carrega nenhuma informação geométrica. Os pixels de imagens diferentes estão misturados numa sequência plana.

**Exemplo**: $P^{(i)}_{r,c}$ representa o pixel na linha $r$ e coluna $c$ da $i$-ésima imagem. O índice $i$ segue de 0 a 59.999, enquanto $r$ e $c$ seguem uma sequência de 0 a 27. Ou seja, a cada 784 pixels (28x28), a estrutura de uma nova imagem começa. $784 \times 60.000 = 47.040.000$ pixels no total.

---

## Etapa 3: Reshape, reconstrução geométrica

```python
images = images.reshape(len(labels), rows, cols)
```

Esta é a operação central do script do ponto de vista da álgebra linear.

### O que o reshape faz

O método `.reshape()` não altera nenhum valor na memória. Ele apenas reorganiza a forma como os dados são indexados. O vetor unidimensional $\mathbf{v}$ é reinterpretado como um tensor tridimensional $\mathcal{X}$ com as seguintes dimensões:

A transformação pode ser descrita como:

$$\mathbf{v} \in \mathbb{Z}^{47.040.000} \xrightarrow{\text{reshape}(60000,\, 28,\, 28)} \mathcal{X} \in \mathbb{Z}^{60000 \times 28 \times 28}$$

### Estrutura do tensor resultante

O resultado é um tensor tridimensional $\mathcal{X}$, onde cada fatia $\mathcal{X}[i]$ é uma matriz $28 \times 28$ correspondente à $i$-ésima imagem:

$$\mathcal{X}[i] = \begin{bmatrix} P_{1,1} & P_{1,2} & \cdots & P_{1,28} \\ P_{2,1} & P_{2,2} & \cdots & P_{2,28} \\ \vdots & \vdots & \ddots & \vdots \\ P_{28,1} & P_{28,2} & \cdots & P_{28,28} \end{bmatrix} \in \mathbb{Z}^{28 \times 28}$$

O tensor completo empilha todas as 60.000 matrizes:

$$\mathcal{X} = \begin{bmatrix} \mathcal{X}[0] \\ \mathcal{X}[1] \\ \vdots \\ \mathcal{X}[59999] \end{bmatrix}, \quad \mathcal{X} \in \mathbb{Z}^{60000 \times 28 \times 28}$$

### Correspondência entre rótulo e imagem

O índice $i$ é compartilhado entre $\mathcal{X}$ e $\mathbf{y}_{\text{train}}$:

$$\mathcal{X}[i] \longleftrightarrow y_i$$

Ou seja, a imagem na posição $i$ do tensor possui o dígito correto armazenado na posição $i$ do vetor de rótulos.

**Exemplo**: Se $y_{42} = 7$, então $\mathcal{X}[42]$ é a matriz de pixels que representa o dígito 7.

Verificação do formato:

```
x_train.shape → (60000, 28, 28)
y_train.shape → (60000,)
```
