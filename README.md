<div align="center">

# Robust MNIST

Um framework de estresse computacional que audita a resiliência e a capacidade de generalização de redes neurais através de perturbações matriciais.

![Status](https://img.shields.io/badge/Status-Em%20Desenvolvimento-blue)
![Python](https://img.shields.io/badge/Python-v3.12.3-blue?logo=python&logoColor=white)

</div>

**Robust MNIST** é um framework de avaliação diagnóstica projetado para testar os limites de redes neurais e modelos de Machine Learning. Através do estresse controlado por perturbações matriciais geométricas e estocásticas (MNIST-C), o algoritmo audita a capacidade de generalização e resiliência desses classificadores, revelando a fragilidade oculta de modelos que atingem acurácia quase perfeita em ambientes controlados.

## Funcionalidades

- **Perturbações Matriciais Avançadas**: Aplicação de transformações geométricas (matrizes de rotação) e estocásticas (convoluções lineares para filtros de ruído e blur) baseadas no ecossistema MNIST-C.
- **Pipeline Algorítmico Nativo**: Decodificação manual de arquivos binários `.ubyte` e manipulação de tensores via NumPy, sem dependência de carregadores automáticos de alto nível.
- **Avaliação Diagnóstica de Robustez**: Cálculo do limite de quebra (*breaking point*), análise do delta ($\Delta$) de queda de acurácia e geração de matrizes de confusão $10 \times 10$ para mapeamento de falhas.

## Instalação e Execução

1. Clone o repositório:

```bash
git clone https://github.com/eukauacavalcante/robust-mnist.git
cd robust-mnist
```

2. Crie e ative o seu ambiente virtual (`venv`):

```bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

3. Instale as dependências do projeto:

```bash
pip install -r requirements.txt
```

4. Execute o pipeline principal:

```bash
python main.py
```

## Sumário da Documentação

- [Pipeline de Dados (Dataset)](docs/dataset.md): Descreve a decodificação dos arquivos binários `.ubyte` e a reestruturação dos dados em tensores tridimensionais dentro de `src/dataset.py`.
- [Pré-processamento das Imagens](docs/preprocessing.md): Realiza a normalização e transformação das imagens para o formato adequado.
