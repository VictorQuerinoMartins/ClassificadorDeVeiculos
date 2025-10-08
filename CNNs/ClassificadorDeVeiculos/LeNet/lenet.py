# -*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Aponta para a pasta do dataset no seu computador.
base_dir = r'C:\Users\yguin\OneDrive\Documentos\GitHub\UNESPAR\IA\CNNs\ClassificadorDeVeiculos\dataset'

if not os.path.exists(base_dir):
    raise FileNotFoundError(f"ERRO: O caminho '{base_dir}' nao foi encontrado. Verifique se esta correto.")
else:
    print(f"Dataset encontrado em: {base_dir}")

# Parâmetros para as imagens e o treinamento
IMG_SIZE = (96, 96)
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.2

# Cria o conjunto de dados para TREINAMENTO (80% dos dados)
train_ds = tf.keras.utils.image_dataset_from_directory(
    base_dir,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=123,  # 'seed' garante que a divisão seja sempre a mesma para reprodutibilidade.
    label_mode='int',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Cria o conjunto de dados para VALIDAÇÃO (20% dos dados)
validation_ds = tf.keras.utils.image_dataset_from_directory(
    base_dir,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=123,  # O 'seed' DEVE ser o mesmo para não haver sobreposição de dados.
    label_mode='int',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)
print("Classes carregadas pelo TensorFlow:", class_names)
print("-" * 30)

# Otimiza os datasets para um carregamento mais rápido durante o treinamento.
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Arquitetura da Rede Neural Convolucional (LeNet-5 com ativação ReLU)
model = tf.keras.Sequential([
    # Normaliza os pixels da imagem para o intervalo [0, 1].
    tf.keras.layers.Rescaling(1./255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    
    # 1ª Convolução: Extrai 6 tipos de características da imagem.
    tf.keras.layers.Conv2D(6, kernel_size=(5, 5), activation='relu', padding='same'),
    # 1º Pooling: Reduz o tamanho da imagem, mantendo as características importantes.
    tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
    
    # 2ª Convolução: Extrai 16 características mais complexas.
    tf.keras.layers.Conv2D(16, kernel_size=(5, 5), activation='relu'),
    # 2º Pooling: Reduz a dimensionalidade novamente.
    tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
    
    # Flatten: Transforma a matriz de características em um vetor único.
    tf.keras.layers.Flatten(),
    
    # 1ª Camada Densa: Combina as características para aprender padrões (120 neurônios).
    tf.keras.layers.Dense(120, activation='relu'),
    # 2ª Camada Densa: Refina os padrões aprendidos (84 neurônios).
    tf.keras.layers.Dense(84, activation='relu'),
    
    # Camada de Saída: Classifica a imagem na categoria mais provável.
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])


# <<< ALTERAÇÃO AQUI: Definindo explicitamente a taxa de aprendizado >>>
# Cria o otimizador Adam com a taxa de aprendizado desejada.
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# Compila o modelo, passando o otimizador criado.
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
print("-" * 30)

# Treina o modelo por um número definido de épocas.
epochs = 21
print(f"Iniciando o treinamento por {epochs} épocas...")
history = model.fit(train_ds, validation_data=validation_ds, epochs=epochs)
print("Treinamento finalizado!")

# Calcula a perda e acurácia médias em todo o conjunto de validação.
print("\nAvaliando o desempenho final do modelo...")
final_loss, final_accuracy = model.evaluate(validation_ds, verbose=0)

# Imprime os resultados formatados no terminal.
print("-" * 30)
print(f"Acurácia Final de Validação: {final_accuracy:.4f} (ou {final_accuracy*100:.2f}%)")
print(f"Perda Final de Validação:   {final_loss:.4f}")
print("-" * 30)

# Gera e exibe os gráficos de Acurácia e Perda ao longo das épocas.
print("\nGerando gráficos de Acurácia e Perda...")
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)

plt.figure(figsize=(12, 5))
plt.suptitle(f"Resultados do Treinamento\nAcurácia Final de Validação: {final_accuracy*100:.2f}%", fontsize=16)

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Acurácia de Treino')
plt.plot(epochs_range, val_acc, label='Acurácia de Validação')
plt.legend(loc='lower right')
plt.title('Acurácia de Treino e Validação')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Perda de Treino')
plt.plot(epochs_range, val_loss, label='Perda de Validação')
plt.legend(loc='upper right')
plt.title('Curva de Perda (Loss)')
plt.show()

# Gera e exibe a Matriz de Confusão para avaliar o desempenho em detalhes.
print("\nGerando Matriz de Confusão...")
y_true = []
y_pred = []

for images, labels in validation_ds:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

# Força a matriz a ter o tamanho correto (ex: 3x3) para evitar erros
# caso uma classe não apareça no conjunto de teste.
cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
ax.set_title(f"Matriz de Confusão\nAcurácia Final: {final_accuracy*100:.2f}%")
plt.show()