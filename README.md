# Classificador de Veículos com CNN

## Visão Geral

[cite_start]Este projeto utiliza Redes Neurais Convolucionais (CNNs) para a classificação de imagens de veículos, categorizando-os em carros, caminhões e vans[cite: 8]. [cite_start]Trata-se de um problema de classificação multiclasse que aproveita a capacidade das CNNs de extrair características complexas de dados visuais[cite: 9].

As aplicações práticas deste classificador incluem:
* [cite_start]**Gestão de Trânsito**: Otimizar o fluxo de veículos, fiscalizar zonas de restrição e auxiliar no planejamento de manutenção de vias[cite: 15].
* [cite_start]**Segurança**: Acelerar a resposta a emergências, informando às equipes de resgate os tipos de veículos envolvidos em acidentes[cite: 17, 18, 19, 20, 21].

## Dataset

[cite_start]Para o treinamento dos modelos, foi utilizado um dataset de imagens de veículos obtido na plataforma Kaggle[cite: 24, 25, 27]. [cite_start]O dataset contém uma variedade de imagens de carros, caminhões e vans, permitindo que o modelo aprenda a distinguir as características visuais de cada classe[cite: 24].

## Arquiteturas e Resultados

Foram avaliadas três arquiteturas de CNN: LeNet, AlexNet e um Modelo Livre customizado. O desempenho foi medido pela acurácia e pela perda (*loss*).

### Comparação de Desempenho

| Modelo | Acurácia | Perda (Loss) |
| :--- | :--- | :--- |
| **Modelo Livre** | [cite_start]**98,63%** [cite: 481] | [cite_start]0.1967 [cite: 481] |
| LeNet | [cite_start]96,13% [cite: 482] | [cite_start]0.1989 [cite: 482] |
| AlexNet | [cite_start]95,83% [cite: 483] | [cite_start]0.1179 [cite: 483] |

[cite_start]O **Modelo Livre** apresentou o melhor desempenho, pois sua arquitetura foi mais eficaz em capturar as características distintas entre os tipos de veículos[cite: 486].

### Otimização de Hiperparâmetros (Auto-Tuning)

[cite_start]Foi realizado um processo de ajuste fino dos hiperparâmetros para otimizar o desempenho da rede LeNet, resultando em uma melhora da acurácia de 96,13% para 98,15%[cite: 423].

* [cite_start]**Taxa de Aprendizagem (Learning Rate)**: O valor de **0.001** foi considerado o ideal, encontrando um bom equilíbrio de aprendizado[cite: 246]. [cite_start]Taxas de 0.01 levaram a overfitting [cite: 242, 243] [cite_start]e 0.0001 resultaram em um aprendizado muito lento[cite: 249].
* [cite_start]**Tamanho do Lote (Batch Size)**: O tamanho de **16** apresentou o melhor resultado, com maior acurácia e menor perda[cite: 374]. [cite_start]Um batch size de 32 mostrou instabilidade [cite: 375] [cite_start]e o de 64 foi o mais fraco[cite: 377].
* [cite_start]**Épocas (Epochs)**: O número ideal de épocas foi definido como **21**[cite: 416]. [cite_start]Após 22 épocas, o modelo começou a apresentar overfitting, com a perda de validação aumentando[cite: 418].

### Arquitetura do Modelo Livre (Vencedor)

[cite_start]O modelo foi treinado com imagens de entrada de dimensão 96x96 pixels com 3 canais de cor (RGB)[cite: 459]. A arquitetura detalhada é a seguinte:

1.  [cite_start]**Conv2D + ReLU**: Extrai 32 características (textura, borda) - Saída: 96x96x32[cite: 461].
2.  [cite_start]**Conv2D + ReLU**: Refina as 32 características - Saída: 96x96x32[cite: 464].
3.  [cite_start]**MaxPooling2D**: Reduz as dimensões pela metade - Saída: 48x48x32[cite: 466].
4.  [cite_start]**Conv2D + ReLU**: Extrai 64 características mais complexas - Saída: 48x48x64[cite: 467].
5.  [cite_start]**Conv2D + ReLU**: Refina as 64 características - Saída: 48x48x64 (Nota: A fonte indica 18x48x64, mas 48x48x64 é o formato esperado)[cite: 470].
6.  [cite_start]**MaxPooling2D**: Reduz as dimensões novamente - Saída: 24x24x64[cite: 470].
7.  [cite_start]**Conv2D + ReLU**: Extrai 128 características abstratas - Saída: 24x24x128[cite: 472].
8.  [cite_start]**MaxPooling2D**: Última redução de dimensionalidade - Saída: 12x12x128[cite: 477].
9.  [cite_start]**Flatten**: Transforma a matriz em um vetor único - Saída: 18.432 neurônios[cite: 477].
10. [cite_start]**Dense + ReLU**: Inicia a classificação - Saída: 256 neurônios[cite: 477].
11. [cite_start]**Dropout (50%)**: Zera 50% dos neurônios para evitar overfitting - Saída: 256 neurônios[cite: 475].
12. [cite_start]**Dense + Softmax**: Gera a probabilidade final para cada uma das 3 classes[cite: 476].

## Conclusão

[cite_start]O Modelo Livre demonstrou superioridade devido à sua arquitetura mais complexa, que permitiu uma representação interna mais rica das imagens e, consequentemente, menos erros de classificação[cite: 487]. [cite_start]O estudo também mostrou que modelos mais simples, como a LeNet, ainda podem ser altamente competitivos quando seus hiperparâmetros são devidamente ajustados[cite: 488].

## Autores

* [cite_start]Gabriel Ricetto [cite: 4]
* [cite_start]João Vitor Pastori [cite: 4]
* [cite_start]Victor Querino Martins [cite: 4]
