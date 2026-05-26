# Explicação: `src/model.py`

Este documento apresenta a especificação matemática, a estrutura de álgebra linear e o fluxo lógico de execução do modelo de rede neural implementado em `src/model.py`. O sistema consiste em uma rede totalmente conectada (Fully Connected) projetada para a classificação de dígitos manuscritos do dataset MNIST.

---

## 1. Visão Geral da Arquitetura e Fluxo de Dados

O modelo processa os dados de forma sequencial através de uma cadeia de transformações lineares e não-lineares. A estrutura macro do pipeline para cada camada oculta segue o fluxo:

$$\text{Entrada} \longrightarrow \text{Transformação Linear (Dense)} \longrightarrow \text{Batch Normalization}\longrightarrow \text{Ativação (ReLU)} \longrightarrow \text{Regularização (Dropout)} \longrightarrow \text{Saída}$$

O modelo padrão é configurado com a seguinte distribuição de dimensões:

- **Camada de Entrada**: Vetor achatado de pixels com dimensão $784$ (decorrente da imagem original de $28 \times 28$).
- **Camadas Ocultas**: Três blocos sequenciais com $512$, $256$ e $128$ neurônios, respectivamente.
- **Camada de Saída**: Camada densa com $10$ neurônios associada à função de ativação Softmax para representação probabilística das classes (dígitos de 0 a 9).

---

## 2. Inicialização dos Parâmetros

Antes do início do processo de otimização, os parâmetros da rede precisam ser instanciados de modo a evitar a simetria de gradientes.

### Matriz de Pesos ($W$)

Os pesos de cada camada são inicializados através do método Glorot Uniform (também conhecido como Inicialização Xavier). Os valores são amostrados de uma distribuição uniforme dentro de um intervalo simétrico:

$$W_{ij} \sim \mathcal{U}\left(-\sqrt{\frac{6}{n + m}}, \; +\sqrt{\frac{6}{n + m}}\right)$$

- $W_{ij}$: Peso individual que conecta o neurônio $j$ da camada anterior ao neurônio $i$ da camada atual.
- $\mathcal{U}$: Distribuição uniforme contínua.
- $n$: Número de conexões de entrada da camada (unidades da camada anterior).
- $m$: Número de unidades (neurônios) da camada atual.

Este intervalo é deduzido estatisticamente para garantir que a variância dos sinais de ativação e dos gradientes permaneça estável entre as camadas, prevenindo fenômenos de explosão ou desaparecimento do gradiente.

### Vetor de Bias ($b$)

O bias de cada neurônio atua como um limiar de sensibilidade independente das entradas e é inicializado em zero:

$$b = \mathbf{0}$$

---

## 3. Execução Sequencial das Camadas (Forward Pass)

O processamento de um lote de dados (mini-batch) contendo $N$ amostras ocorre através da execução ordenada dos seguintes operadores matemáticos em cada camada da rede.

### Etapa 3.1: Transformação Linear

Cada neurônio realiza a combinação linear ponderada de suas entradas. Para um único vetor de entrada $x$, a operação é definida por:

$$z = Wx + b$$

Para otimizar o processamento em hardware paralelo, o Keras processa simultaneamente um lote de $N$ amostras. A operação adota a formulação matricial:

$$Z = XW^T + \mathbf{1}_{N}b^T$$

- $Z$: Matriz de pré-ativação linear, com dimensão $N \times m$.
- $X$: Matriz de dados de entrada do lote, com dimensão $N \times n$.
- $W^T$: Transposta da matriz de pesos da camada (dimensão $n \times m$).
- $\mathbf{1}_{N}$: Vetor de uns com dimensão $N \times 1$, utilizado para transmitir o vetor de bias $b^T$ (dimensão $1 \times m$) para todas as $N$ linhas da matriz do lote.

### Etapa 3.2: Batch Normalization

Após a transformação linear, a Batch Normalization estabiliza a distribuição das pré-ativações $Z$ ao longo do lote. Para cada característica (neurônio) $j$ dentro do mini-batch, o algoritmo computa:

1. Média do lote:

$$\mu_j = \frac{1}{N} \sum_{i=1}^{N} z_{ij}$$


2. Variância do lote:

$$\sigma^2_j = \frac{1}{N} \sum_{i=1}^{N} (z_{ij} - \mu_j)^2$$


3. Normalização:

$$\hat{z}_{ij} = \frac{z_{ij} - \mu_j}{\sqrt{\sigma^2_j + \varepsilon}}$$


4. Escalonamento linear:

$$\text{BN}(z_{ij}) = \gamma_j \hat{z}_{ij} + \beta_j$$



- $z_{ij}$: Valor de pré-ativação da amostra $i$ para o neurônio $j$.
- $\mu_j$: Média aritmética dos valores do neurônio $j$ no lote atual.
- $\sigma^2_j$: Variância dos valores do neurônio $j$ no lote atual.
- $\varepsilon$: Constante numérica de estabilização (configurada em $10^{-5}$) para evitar divisões por zero.
- $\hat{z}_{ij}$: Valor normalizado com média zero e variância unitária.
- $\gamma_j, \beta_j$: Parâmetros escalares treináveis de escala e deslocamento, que permitem à rede restaurar a capacidade de representação linear original se necessário.

### Etapa 3.3: Ativação Não-Linear (ReLU)

A função de ativação introduz propriedades não-lineares no espaço de representação, permitindo a modelagem de fronteiras de decisão complexas. Aplica-se a função retificadora linear unitária (ReLU) elemento a elemento sobre a matriz normalizada:

$$\text{ReLU}(z) = \max(0, z)$$

Se o valor normalizado for negativo ou nulo, a saída é fixada em zero; se for positivo, o valor é integralmente preservado. Isso gera uma ativação esparsa na rede e mitiga o problema de desvanecimento de gradientes em regiões saturadas.

### Etapa 3.4: Regularização Estocástica (Dropout)

Para mitigar o sobreajuste (overfitting), aplica-se uma máscara binária baseada em uma distribuição de Bernoulli com probabilidade de sobrevivência $1-p$, onde $p$ é a taxa de dropout:

$$a_i^{\text{drop}} = a_i \cdot m_i, \quad m_i \sim \text{Bernoulli}(1 - p)$$

- $a_i$: Ativação original do neurônio $i$ após a ReLU.
- $m_i$: Variável aleatória binária que assume valor $0$ com probabilidade $p$ e valor $1$ com probabilidade $1-p$.
- $a_i^{\text{drop}}$: Ativação regularizada.

Para manter a consistência do valor esperado do sinal entre as fases de treinamento e teste, o Keras utiliza o método Inverted Dropout, escalando as ativações sobreviventes durante o treino pelo fator $\frac{1}{1-p}$. Na fase de inferência, o Dropout é desativado.

---

## 4. Camada de Saída e Predição

A última camada da rede realiza a classificação probabilística das amostras em 10 classes possíveis. Ela recebe o vetor de características transformadas da última camada oculta e aplica uma transformação linear para gerar os scores brutos (logits) $z_{\text{out}} \in \mathbb{R}^{10}$.

Esses logits são mapeados para uma distribuição de probabilidade por meio da função Softmax:

$$p_i = \frac{e^{z_i}}{\sum_{k=0}^{9} e^{z_k}}, \quad \forall i \in \{0, 1, \ldots, 9\}$$

- $z_i$: Logit associado à classe $i$.
- $e^{z_i}$: Exponenciação do logit para garantir valores estritamente positivos.
- $\sum_{k=0}^{9} e^{z_k}$: Termo de normalização que representa a soma dos logits exponeciados de todas as classes.
- $p_i$: Probabilidade final atribuída à amostra pertencer à classe $i$, garantindo que $p_i \in (0, 1)$ e $\sum_{i=0}^{9} p_i = 1$.

---

## 5. Função de Perda e Regularização Global

A avaliação do erro do modelo durante o treinamento é conduzida pela função de perda de Entropia Cruzada Categórica (Categorical Cross-Entropy). Para um mini-batch de $N$ amostras, a perda baseada nos erros de predição é dada por:

$$L_{\text{CE}} = -\frac{1}{N} \sum_{i=1}^{N} \log\left(p_{i, y_i}\right)$$

- $p_{i, y_i}$: Probabilidade calculada pela Softmax para a classe real $y_i$ da amostra $i$.

Para limitar o crescimento excessivo dos pesos e controlar a complexidade do modelo, adiciona-se uma penalidade de regularização L2 (baseada na Norma de Frobenius ao quadrado) sobre as matrizes de pesos de todas as camadas densas:

$$L_{\text{total}} = L_{\text{CE}} + \lambda \sum_{c} \|W_c\|^2_F$$

- $\lambda$: Coeficiente de regularização L2 (definido como $10^{-4}$), que controla a intensidade da penalização.
- $\|W_c\|^2_F$: Norma de Frobenius ao quadrado da matriz de pesos da camada $c$, calculada pela soma dos quadrados de todos os seus elementos individuais: $\sum_{i,j} (W_{ij})^2$.

> **Nota de implementação**: o Keras recebe os rótulos como inteiros escalares ($y_i \in \{0, \ldots, 9\}$) via `sparse_categorical_crossentropy`, eliminando a necessidade de converter para vetores one-hot de dimensão 10. Matematicamente equivalente, mas economiza memória e a etapa de conversão em tempo de execução.

---

## 6. Mecanismo de Otimização: Algoritmo Adam

A atualização dos parâmetros treináveis (tanto as matrizes de pesos $W$ quanto os vetores de bias $b$) ocorre a cada passo com base no cálculo dos gradientes locais obtidos via retropropagação (Backpropagation). O cálculo do gradiente para uma iteração $t$ é definido como:

$$g_t = \frac{\partial L_{\text{total}}}{\partial W}$$

O otimizador Adam processa esse gradiente utilizando o histórico estatístico dos momentos de primeira e segunda ordem:

1. Média móvel dos gradientes (Momentum):

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$


2. Média móvel dos gradientes ao quadrado (RMSProp):

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$


3. Correção de viés estatístico para as iterações iniciais:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$


4. Equação final de atualização dos parâmetros:

$$W_t = W_{t-1} - \frac{\alpha}{\sqrt{\hat{v}_t} + \varepsilon} \cdot \hat{m}_t$$



- $W_t$: Nova matriz de pesos atualizada após o passo de treinamento.
- $W_{t-1}$: Matriz de pesos do passo anterior.
- $\alpha$: Taxa de aprendizado base (learning rate), configurada em $10^{-3}$.
- $\beta_1, \beta_2$: Coeficientes de decaimento exponencial para as estimativas dos momentos, configurados por padrão em $0.9$ e $0.999$, respectivamente.
- $\hat{m}_t$: Vetor direcional baseado no histórico acumulado de trajetórias, atuando para acelerar a convergência em direções consistentes.
- $\hat{v}_t$: Medida da magnitude das oscilações passadas de cada parâmetro, agindo no denominador como um modulador adaptativo que diminui o passo em componentes hiperativas e aumenta o passo em componentes estáveis.
- $\varepsilon$: Constante de suporte numérico fixa em $10^{-7}$ para impedir indeterminações por divisão por zero.

---

## 7. Estrutura Matemática dos Parâmetros

A quantidade de parâmetros de uma camada densa é uma consequência direta de suas dimensões lineares. Para uma camada com $n$ entradas e $m$ neurônios de saída, o total de variáveis adaptativas é computado por:

$$\text{Parâmetros} = \underbrace{(n \times m)}_{W} + \underbrace{m}_{b} = m(n + 1)$$

O mapeamento exato das dimensões e parâmetros do modelo compilado está detalhado na tabela abaixo:

| Identificador da Camada | Conexões de Entrada ($n$) | Unidades de Saída ($m$) | Elementos em $W$ | Elementos em $b$ | Total de Parâmetros |
| --- | --- | --- | --- | --- | --- |
| **Dense 1** | 784 | 512 | 401.408 | 512 | 401.920 |
| **Dense 2** | 512 | 256 | 131.072 | 256 | 131.328 |
| **Dense 3** | 256 | 128 | 32.768 | 128 | 32.896 |
| **Dense Saída** | 128 | 10 | 1.280 | 10 | 1.290 |
| **Subtotal Dense** | — | — | 566.528 | 906 | **567.434** |

As camadas de Batch Normalization inserem $4 \times m$ parâmetros adicionais por bloco (compostos por $\gamma$, $\beta$, além da média e variância móveis calculadas durante o processo). Para as dimensões configuradas $(512 + 256 + 128)$, adicionam-se $4 \times 896 = 3.584$ parâmetros.

O volume global do modelo resulta em **571.018 parâmetros**.
