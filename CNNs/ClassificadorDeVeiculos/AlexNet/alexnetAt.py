import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os


base_dir = r'C:\Users\yguin\OneDrive\Documentos\GitHub\CNN\CNNs\ClassificadorDeVeiculos\dataset'
if not os.path.exists(base_dir):
    raise FileNotFoundError(f"ERRO: O caminho '{base_dir}' nao foi encontrado.")

IMG_SIZE = (96, 96)        
VALIDATION_SPLIT = 0.2     
SEED = 123
BEST_LR = 0.001           
BEST_BATCH_SIZE = 16       
LONG_EPOCHS = 21          


def create_model(num_classes, learning_rate=0.001):
    model = tf.keras.Sequential([
       
        tf.keras.layers.Rescaling(1./255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)), # Normaliza os pixels 0,1
        
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),           #Encontra 32 padrões
      
        tf.keras.layers.MaxPooling2D(),                                            # Reduz o tamanho da imagem pela metade 

        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),         # Encontra 64 padrões mais complexos
       
        tf.keras.layers.MaxPooling2D(),                                          # Reduz o tamanho do mapa de padrões novamente

        tf.keras.layers.Flatten(),                                              # Junta todos em um vetor (para a parte 'densa')

        tf.keras.layers.Dense(128, activation='relu'),                         # Combina as caracteristicas aprendidas para fazer a classificação
        
        tf.keras.layers.Dense(num_classes, activation='softmax')              # Gera a probabilidade 
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