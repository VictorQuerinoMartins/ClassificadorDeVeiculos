import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

base_dir = r'C:\Users\yguin\OneDrive\Documentos\GitHub\UNESPAR\IA\CNNs\ClassificadorDeVeiculos\dataset'

if not os.path.exists(base_dir):
    raise FileNotFoundError(f"ERRO: O caminho '{base_dir}' nao foi encontrado. Verifique se esta correto.")
else:
    print(f"Dataset encontrado em: {base_dir}")

IMG_SIZE = (96, 96)
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.2

train_ds = tf.keras.utils.image_dataset_from_directory(
    base_dir,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=123, 
    label_mode='int',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

validation_ds = tf.keras.utils.image_dataset_from_directory(
    base_dir,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=123, 
    label_mode='int',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)
print("Classes carregadas pelo TensorFlow:", class_names)
print("-" * 30)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Arquitetura da Rede Neural Convolucional (LeNet-5 com ativacao ReLU)
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    
    tf.keras.layers.Conv2D(6, kernel_size=(5, 5), activation='relu', padding='same'),
    tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
    
    tf.keras.layers.Conv2D(16, kernel_size=(5, 5), activation='relu'),
    tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
    
    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(120, activation='relu'),
    tf.keras.layers.Dense(84, activation='relu'),
    
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])


optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
print("-" * 30)

epochs = 21
print(f"Iniciando o treinamento por {epochs} epocas...")
history = model.fit(train_ds, validation_data=validation_ds, epochs=epochs)
print("Treinamento finalizado!")

print("\nAvaliando o desempenho final do modelo...")
final_loss, final_accuracy = model.evaluate(validation_ds, verbose=0)

print("-" * 30)
print(f"Acuracia Final de Validacao: {final_accuracy:.4f} (ou {final_accuracy*100:.2f}%)")
print(f"Perda Final de Validacao:   {final_loss:.4f}")
print("-" * 30)

print("\nGerando graficos de Acuracia e Perda...")
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)

plt.figure(figsize=(12, 5))
plt.suptitle(f"Resultados do Treinamento\nAcuracia Final de Validacao: {final_accuracy*100:.2f}%", fontsize=16)

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Acuracia de Treino')
plt.plot(epochs_range, val_acc, label='Acuracia de Validacao')
plt.legend(loc='lower right')
plt.title('Acuracia de Treino e Validacao')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Perda de Treino')
plt.plot(epochs_range, val_loss, label='Perda de Validacao')
plt.legend(loc='upper right')
plt.title('Curva de Perda (Loss)')
plt.show()

print("\nGerando Matriz de Confusao...")
y_true = []
y_pred = []

for images, labels in validation_ds:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))


cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
ax.set_title(f"Matriz de Confusao\nAcuracia Final: {final_accuracy*100:.2f}%")
plt.show()