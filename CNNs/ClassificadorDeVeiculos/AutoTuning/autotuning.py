import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIGURAÇÃO BASE ---
base_dir = r'C:\Users\yguin\OneDrive\Documentos\GitHub\UNESPAR\IA\CNNs\ClassificadorDeVeiculos\dataset'
if not os.path.exists(base_dir):
    raise FileNotFoundError(f"ERRO: O caminho '{base_dir}' nao foi encontrado.")


IMG_SIZE = (96, 96)
VALIDATION_SPLIT = 0.2
SEED = 123
FIXED_BATCH_SIZE = 32

FIXED_EPOCHS = 5

def create_model(learning_rate=0.001):
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(class_names), activation='softmax')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_history(history, parameter_name, value):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.figure(figsize=(12, 5))
    plt.suptitle(f'Resultados para {parameter_name} = {value}', fontsize=16)
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

train_ds = tf.keras.utils.image_dataset_from_directory(
    base_dir, validation_split=VALIDATION_SPLIT, subset="training",
    seed=SEED, image_size=IMG_SIZE, batch_size=FIXED_BATCH_SIZE, label_mode='int'
)
validation_ds = tf.keras.utils.image_dataset_from_directory(
    base_dir, validation_split=VALIDATION_SPLIT, subset="validation",
    seed=SEED, image_size=IMG_SIZE, batch_size=FIXED_BATCH_SIZE, label_mode='int'
)
class_names = train_ds.class_names
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)

learning_rates_to_test = [0.01, 0.001, 0.0001] # Teste com taxa de aprendizagem variando tamanho dos passos

for lr in learning_rates_to_test:
    print("-" * 50)
    print(f"INICIANDO TESTE COM LEARNING RATE: {lr}")
    print("-" * 50)
    
    model = create_model(learning_rate=lr)
    history = model.fit(
        train_ds,
        epochs=FIXED_EPOCHS,
        validation_data=validation_ds,
        verbose=1
    )
    plot_history(history, "Learning Rate", lr)

print("Fim do teste de Taxa de Aprendizagem.")