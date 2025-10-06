# -- coding: utf-8 --
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# -----------------------------
# 1. Dataset
# -----------------------------
base_dir = r'C:\Users\yguin\OneDrive\Documentos\GitHub\CNN\CNNs\ClassificadorDeVeiculos\dataset'
if not os.path.exists(base_dir):
    raise FileNotFoundError(f"Diretório não encontrado: {base_dir}")

# Para acelerar, podemos reduzir temporariamente o tamanho da imagem
IMG_SIZE = (128, 128)
VALIDATION_SPLIT = 0.2

train_ds = tf.keras.utils.image_dataset_from_directory(
    base_dir,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=16,
    label_mode='int'
)

validation_ds = tf.keras.utils.image_dataset_from_directory(
    base_dir,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=16,
    label_mode='int',
    shuffle=False
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)
print("Classes:", class_names)

# Otimização do dataset
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)

# -----------------------------
# 2. AlexNet (versão menor para teste Auto-Tuning)
# -----------------------------
def create_alexnet_model(learning_rate=0.0001, small=False):
    if small:
        filters1, filters2, filters3, filters4, filters5 = 32, 64, 64, 64, 32
        dense_units = 512
    else:
        filters1, filters2, filters3, filters4, filters5 = 96, 256, 384, 384, 256
        dense_units = 4096
    
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        tf.keras.layers.Conv2D(filters1, kernel_size=(11,11), strides=4, activation='relu', padding='valid'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=2),
        tf.keras.layers.Conv2D(filters2, kernel_size=(5,5), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=2),
        tf.keras.layers.Conv2D(filters3, kernel_size=(3,3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(filters4, kernel_size=(3,3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(filters5, kernel_size=(3,3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(dense_units, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(dense_units, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# -----------------------------
# 3. Auto-Tuning rápido
# -----------------------------
best_lr = 0.0001
best_batch = 16
best_epochs = 10

# i. Taxa de aprendizado (testar versão pequena)
for lr in [0.001, 0.0005, 0.0001]:
    print(f"\nTestando learning rate: {lr}")
    model = create_alexnet_model(learning_rate=lr, small=True)
    history = model.fit(train_ds, validation_data=validation_ds, epochs=3)  # treino curto

# ii. Tamanho do batch
for bs in [16, 32]:
    print(f"\nTestando batch size: {bs}")
    train_bs = train_ds.batch(bs)
    val_bs = validation_ds.batch(bs)
    model = create_alexnet_model(learning_rate=best_lr, small=True)
    history = model.fit(train_bs, validation_data=val_bs, epochs=3)

# iii. Número de épocas (apenas observação)
for ep in [5, 10]:
    print(f"\nTestando número de épocas: {ep}")
    model = create_alexnet_model(learning_rate=best_lr, small=True)
    history = model.fit(train_ds, validation_data=validation_ds, epochs=ep)

# -----------------------------
# 4. Treinamento final AlexNet
# -----------------------------
# Agora sim usamos a AlexNet completa (grande) com melhores hiperparâmetros
IMG_SIZE = (227, 227)
train_ds = tf.keras.utils.image_dataset_from_directory(
    base_dir, validation_split=VALIDATION_SPLIT, subset="training",
    seed=123, image_size=IMG_SIZE, batch_size=best_batch, label_mode='int'
)
validation_ds = tf.keras.utils.image_dataset_from_directory(
    base_dir, validation_split=VALIDATION_SPLIT, subset="validation",
    seed=123, image_size=IMG_SIZE, batch_size=best_batch, label_mode='int', shuffle=False
)
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)

model = create_alexnet_model(learning_rate=best_lr, small=False)
history = model.fit(train_ds, validation_data=validation_ds, epochs=best_epochs)

# Curvas de acurácia e perda
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(acc, label='Treino')
plt.plot(val_acc, label='Validação')
plt.legend()
plt.title('Acurácia')

plt.subplot(1,2,2)
plt.plot(loss, label='Treino')
plt.plot(val_loss, label='Validação')
plt.legend()
plt.title('Perda')
plt.show()

# Matriz de Confusão
y_true = []
y_pred = []
for images, labels in validation_ds:
    y_true.extend(labels.numpy())
    preds = model.predict(images)
    y_pred.extend(np.argmax(preds, axis=1))
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusão - AlexNet")
plt.show()

# -----------------------------
# 5. Modelo final customizado (livre)
# -----------------------------
def create_custom_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        tf.keras.layers.Conv2D(64, kernel_size=(5,5), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.Conv2D(128, kernel_size=(5,5), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_lr),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

custom_model = create_custom_model()
history_custom = custom_model.fit(train_ds, validation_data=validation_ds, epochs=best_epochs)