# Classificador de Veículos com CNN

## Visão Geral

Este projeto utiliza Redes Neurais Convolucionais (CNNs) para a classificação de imagens de veículos, categorizando-os em carros, caminhões e vans. Trata-se de um problema de classificação multiclasse que aproveita a capacidade das CNNs de extrair características complexas de dados visuais.

As aplicações práticas deste classificador incluem:
* **Gestão de Trânsito**: Otimizar o fluxo de veículos, fiscalizar zonas de restrição e auxiliar no planejamento de manutenção de vias.
* **Segurança**: Acelerar a resposta a emergências, informando às equipes de resgate os tipos de veículos envolvidos em acidentes.

## Dataset

Para o treinamento dos modelos, foi utilizado um dataset de imagens de veículos obtido na plataforma Kaggle. O dataset contém uma variedade de imagens de carros, caminhões e vans, permitindo que o modelo aprenda a distinguir as características visuais de cada classe.

## Arquiteturas e Resultados

Foram avaliadas três arquiteturas de CNN: LeNet, AlexNet e um Modelo Livre customizado. O desempenho foi medido pela acurácia e pela perda (*loss*).

### Comparação de Desempenho

| Modelo | Acurácia | Perda (Loss) |
| :--- | :--- | :--- |
| **Modelo Livre** | **98,63%** | 0.1967 |
| LeNet | 96,13% | 0.1989 |
| AlexNet | 95,83% | 0.1179 |

O **Modelo Livre** apresentou o melhor desempenho, pois sua arquitetura foi mais eficaz em capturar as características distintas entre os tipos de veículos.

### Otimização de Hiperparâmetros (Auto-Tuning)

Foi realizado um processo de ajuste fino dos hiperparâmetros para otimizar o desempenho da rede LeNet, resultando em uma melhora da acurácia de 96,13% para 98,15%.

* **Taxa de Aprendizagem (Learning Rate)**: O valor de **0.001** foi considerado o ideal, encontrando um bom equilíbrio de aprendizado. Taxas de 0.01 levaram a overfitting e 0.0001 resultaram em um aprendizado muito lento.
* **Tamanho do Lote (Batch Size)**: O tamanho de **16** apresentou o melhor resultado, com maior acurácia e menor perda. Um batch size de 32 mostrou instabilidade e o de 64 foi o mais fraco.
* **Épocas (Epochs)**: O número ideal de épocas foi definido como **21**. Após 22 épocas, o modelo começou a apresentar overfitting, com a perda de validação aumentando.

### Arquitetura do Modelo Livre (Vencedor)

O modelo foi treinado com imagens de entrada de dimensão 96x96 pixels com 3 canais de cor (RGB). A arquitetura detalhada é a seguinte:

1.  **Conv2D + ReLU**: Extrai 32 características (textura, borda) - Saída: 96x96x32.
2.  **Conv2D + ReLU**: Refina as 32 características - Saída: 96x96x32.
3.  **MaxPooling2D**: Reduz as dimensões pela metade - Saída: 48x48x32.
4.  **Conv2D + ReLU**: Extrai 64 características mais complexas - Saída: 48x48x64.
5.  **Conv2D + ReLU**: Refina as 64 características - Saída: 48x48x64.
6.  **MaxPooling2D**: Reduz as dimensões novamente - Saída: 24x24x64.
7.  **Conv2D + ReLU**: Extrai 128 características abstratas - Saída: 24x24x128.
8.  **MaxPooling2D**: Última redução de dimensionalidade - Saída: 12x12x128.
9.  **Flatten**: Transforma a matriz em um vetor único - Saída: 18.432 neurônios.
10. **Dense + ReLU**: Inicia a classificação - Saída: 256 neurônios.
11. **Dropout (50%)**: Zera 50% dos neurônios para evitar overfitting - Saída: 256 neurônios.
12. **Dense + Softmax**: Gera a probabilidade final para cada uma das 3 classes.

## Conclusão

O Modelo Livre demonstrou superioridade devido à sua arquitetura mais complexa, que permitiu uma representação interna mais rica das imagens e, consequentemente, menos erros de classificação. O estudo também mostrou que modelos mais simples, como a LeNet, ainda podem ser altamente competitivos quando seus hiperparâmetros são devidamente ajustados.

## Autores

* Gabriel Ricetto
* João Vitor Pastori
* Victor Querino Martins
