import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

base_dir = r'C:\Users\yguin\OneDrive\Documentos\GitHub\UNESPAR\IA\CNNs\ClassificadorDeVeiculos\dataset'
if not os.path.exists(base_dir):
    raise FileNotFoundError(f"ERRO: O caminho '{base_dir}' nao foi encontrado.")

IMG_SIZE = (96, 96)
VALIDATION_SPLIT = 0.2
SEED = 123
LEARNING_RATE = 0.001
BATCH_SIZE = 16      
EPOCHS = 21          

train_ds = tf.keras.utils.image_dataset_from_directory(
    base_dir, validation_split=VALIDATION_SPLIT, subset="training",
    seed=SEED, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='int'
)
validation_ds = tf.keras.utils.image_dataset_from_directory(
    base_dir, validation_split=VALIDATION_SPLIT, subset="validation",
    seed=SEED, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='int'
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)


def create_free_model(num_classes, learning_rate):
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),  #  Extrai 32 caaracteristicas
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),  # Aprofunda a extracao de caracteristicas
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),                         # Reduz o tamanho da imagem
        
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),  # 64 filtros para caracteristicas mais complexas
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),  # Aprofunda a extracao
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),                         # Reduz o tamanho denovo

        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),           # 128 filtros para caracteristicas complexas
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),                                  # Reduz o tamanho final
        
        tf.keras.layers.Flatten(),                                                       # Transforma a matriz em um vetor 
        tf.keras.layers.Dense(256, activation='relu'),                                   # 256 neurônios para combinar as caracteristicas
        tf.keras.layers.Dropout(0.5),                                                    # Zera 50% dos neurônios para evitar overfitting
        tf.keras.layers.Dense(num_classes, activation='softmax')                         # Probabilidade para cada classe (Carro, Caminhao, Van)
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_history(history, title, final_accuracy):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(14, 6))
    plt.suptitle(f"{title}\nAcurdacia Final de Validacao: {final_accuracy*100:.2f}%", fontsize=16)
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Acurdacia de Treino')
    plt.plot(epochs_range, val_acc, label='Acurdacia de Validacao')
    plt.legend(loc='lower right')
    plt.title('Acurdacia')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Perda de Treino')
    plt.plot(epochs_range, val_loss, label='Perda de Validacao')
    plt.legend(loc='upper right')
    plt.title('Perda (Loss)')
    plt.show()

print("-" * 50)
print(f"INICIANDO TREINO DO MODELO LIVRE")
print(f"Usando LR={LEARNING_RATE}, Batch Size={BATCH_SIZE} e {EPOCHS} épocas")
print("-" * 50)

model = create_free_model(NUM_CLASSES, learning_rate=LEARNING_RATE)
model.summary()

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=validation_ds,
    verbose=1
)
print("Treinamento finalizado!")

final_loss, final_accuracy = model.evaluate(validation_ds, verbose=0)
print("-" * 30)
print(f"Acurdacia Final de Validacao: {final_accuracy:.4f} (ou {final_accuracy*100:.2f}%)")
print(f"Perda Final de Validacao:   {final_loss:.4f}")
print("-" * 30)

plot_history(history, "Resultados do Modelo Livre", final_accuracy)

print("\nGerando a Matriz de Confusao...")
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
ax.set_title(f"Matriz de Confusao - Modelo Livre\nAcurdacia Final: {final_accuracy*100:.2f}%")
plt.show()