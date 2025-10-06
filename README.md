# Classificador de Veículos com CNN

## Descrição

Este projeto implementa um classificador de imagens de veículos usando uma Rede Neural Convolucional (CNN) construída com TensorFlow/Keras. O modelo é treinado para identificar diferentes classes de veículos a partir de um dataset de imagens organizado em subpastas (uma por classe).

O script realiza as seguintes etapas principais:
- Carrega e pré-processa o dataset de imagens (redimensiona para 150x150 pixels).
- Divide os dados em conjuntos de treinamento (80%) e teste/validação (20%).
- Constrói uma CNN simples com camadas convolucionais, pooling, flatten e densas.
- Treina o modelo por 15 épocas usando o otimizador Adam e perda de entropia cruzada esparsa.
- Avalia o modelo gerando e exibindo uma matriz de confusão para visualizar o desempenho.

O modelo é otimizado para desempenho com cache e prefetch no dataset. Ao final, uma matriz de confusão é plotada usando Matplotlib e Scikit-learn.

**Exemplo de uso esperado:** Classificação de veículos como carros, motos, baseado em um dataset como o usado no projeto.

- Python 3.8 ou superior.
- Bibliotecas necessárias:
  - TensorFlow (versão 2.10+ recomendada para suporte a datasets de imagens).
  - Matplotlib (para visualizações).
  - NumPy (para manipulação de arrays).
  - Scikit-learn (para matriz de confusão).
  - OS (biblioteca padrão do Python).

Instale as dependências via pip:
```bash
pip install tensorflow matplotlib numpy scikit-learn
