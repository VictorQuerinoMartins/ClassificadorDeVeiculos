# -- coding: utf-8 --

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

base_dir = r'C:\Users\yguin\OneDrive\Documentos\GitHub\CNN\CNNs\ClassificadorDeVeiculos\dataset'
if not os.path.exists(base_dir):
    raise FileNotFoundError(
        f"O diretório especificado não foi encontrado: {base_dir}\n"
        f"Verifique se as pastas das classes (Carro, Caminhao, Van) estão dentro dele."
    )

# 2. Criar datasets de treino e validação AUTOMATICAMENTE
IMG_SIZE = (227, 227)  # AlexNet original usa 227x227
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2# -*- coding: utf-8 -*-
# ARQUIVO 3: ENCONTRANDO O NÚMERO IDEAL DE ÉPOCAS

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIGURAÇÃO BASE ---
base_dir = r'C:\Users\yguin\OneDrive\Documentos\GitHub\CNN\CNNs\ClassificadorDeVeiculos\dataset'
if not os.path.exists(base_dir):
    raise FileNotFoundError(f"ERRO: O caminho '{base_dir}' nao foi encontrado.")

IMG_SIZE = (96, 96) # <<< MUDANÇA AQUI
VALIDATION_SPLIT = 0.2
SEED = 123
BEST_LR = 0.001
BEST_BATCH_SIZE = 16
LONG_EPOCHS = 50

# --- FUNÇÕES AUXILIARES ---
def create_model(num_classes, learning_rate=0.001):
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_history(history, title):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.figure(figsize=(12, 5))
    plt.suptitle(title, fontsize=16)
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Acurácia de Treino')
    plt.plot(val_acc, label='Acurácia de Validação')
    plt.legend(loc='lower right')
    plt.title('Acurácia')
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Perda de Treino')
    plt.plot(val_loss, label='Perda de Validação')
    plt.legend(loc='upper right')
    plt.title('Perda (Loss)')
    plt.show()

# --- CARREGAMENTO DO DATASET ---
train_ds = tf.keras.utils.image_dataset_from_directory(
    base_dir, validation_split=VALIDATION_SPLIT, subset="training",
    seed=SEED, image_size=IMG_SIZE, batch_size=BEST_BATCH_SIZE, label_mode='int'
)
validation_ds = tf.keras.utils.image_dataset_from_directory(
    base_dir, validation_split=VALIDATION_SPLIT, subset="validation",
    seed=SEED, image_size=IMG_SIZE, batch_size=BEST_BATCH_SIZE, label_mode='int'
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- TREINAMENTO FINAL ---
print("-" * 50)
print(f"INICIANDO TREINO LONGO PARA ACHAR O PONTO DE PARADA IDEAL")
print(f"Usando LR={BEST_LR} e Batch Size={BEST_BATCH_SIZE}")
print("-" * 50)

model = create_model(NUM_CLASSES, learning_rate=BEST_LR)
history = model.fit(
    train_ds,
    epochs=LONG_EPOCHS,
    validation_data=validation_ds,
    verbose=1
)

plot_history(history, f"Treino Longo com Melhores Parâmetros (LR={BEST_LR}, BS={BEST_BATCH_SIZE})")

print("Fim do teste de Épocas.")

train_ds = tf.keras.utils.image_dataset_from_directory(
    base_dir,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'
)

validation_ds = tf.keras.utils.image_dataset_from_directory(
    base_dir,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int',
    shuffle=False
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)
print("\nClasses encontradas:", class_names)
print(f"Número de classes: {NUM_CLASSES}")

# 3. Mostrar 1 exemplo de cada classe
plt.figure(figsize=(10, 5))
for images, labels in train_ds.take(1):
    for i, class_name in enumerate(class_names):
        idx_list = np.where(labels.numpy() == i)[0]
        if len(idx_list) > 0:
            idx = idx_list[0]
            plt.subplot(1, NUM_CLASSES, i + 1)
            plt.imshow(images[idx].numpy().astype("uint8"))
            plt.title(class_name)
            plt.axis("off")
plt.suptitle("Exemplos do Dataset")
plt.show()

# 4. Otimizar dataset para desempenho
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 5. Criar modelo AlexNet padrão
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    
    # 1ª Convolução + ReLU + MaxPooling
    tf.keras.layers.Conv2D(96, kernel_size=(11,11), strides=4, activation='relu', padding='valid'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=2),
    
    # 2ª Convolução + ReLU + MaxPooling
    tf.keras.layers.Conv2D(256, kernel_size=(5,5), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=2),
    
    # 3ª Convolução + ReLU
    tf.keras.layers.Conv2D(384, kernel_size=(3,3), activation='relu', padding='same'),
    
    # 4ª Convolução + ReLU
    tf.keras.layers.Conv2D(384, kernel_size=(3,3), activation='relu', padding='same'),
    
    # 5ª Convolução + ReLU + MaxPooling
    tf.keras.layers.Conv2D(256, kernel_size=(3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=2),
    
    # Flatten + Dense layers
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.summary()

# Configurar otimizador
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 6. Treinar modelo
print("\nIniciando o treinamento do modelo AlexNet...")
history = model.fit(train_ds, validation_data=validation_ds, epochs=20)
print("Treinamento finalizado.")

# 7. Plotar Curva de Perda e Acurácia
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(12, 5))
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

# 8. Gerar Matriz de Confusão
print("\nGerando Matriz de Confusão...")
y_true = []
y_pred = []

for images, labels in validation_ds:
    y_true.extend(labels.numpy())
    preds = model.predict(images)
    y_pred.extend(np.argmax(preds, axis=1))

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusão - Veículos")
plt.show()