
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import RandomFlip, RandomRotation,\
    RandomBrightness
import numpy as np

#█▀█ ▄▀█ █▀ █▀█   ▄█ ▀   █▀█ █▄▄ ▀█▀ █▀▀ █▄░█ █▀▀ █▀█   █▄█   ▀█▀ █▀█ ▄▀█ █▄░█ █▀ █▀▀ █▀█ █▀█ █▀▄▀█ ▄▀█ █▀█
#█▀▀ █▀█ ▄█ █▄█   ░█ ▄   █▄█ █▄█ ░█░ ██▄ █░▀█ ██▄ █▀▄   ░█░   ░█░ █▀▄ █▀█ █░▀█ ▄█ █▀░ █▄█ █▀▄ █░▀░█ █▀█ █▀▄

#█▀▄ ▄▀█ ▀█▀ █▀█ █▀
#█▄▀ █▀█ ░█░ █▄█ ▄█


# Cargamos los datos
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

# Normalizamos las imágenes
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Dimensiones de los conjuntos de datos
print(f'Tamaño del conjunto de entrenamiento: {len(X_train)}')
print(f'Tamaño del conjunto de test: {len(X_test)}')
print(f'Tamaño de las imágenes: {X_train.shape[1:]}')

# Obtenemos dos imágenes del conjunto de entrenamiento para mostrarlas
imagenes, etiquetas = X_train[:2], Y_train[:2]

# Diccionario de etiquetas
clases = {0: 'avión', 1: 'coche', 2: 'pájaro', 3: 'gato', 4: 'ciervo',\
          5: 'perro', 6: 'rana', 7: 'caballo', 8: 'barco', 9: 'camión'}

# Visualizamos las imágenes y etiquetas de un batch de entrenamiento
fig, axs = plt.subplots(1, 2, figsize=(10, 10))
for i in range(2):
    axs[i].imshow(imagenes[i])
    axs[i].set_title(f'Etiqueta: {clases[etiquetas[i].item()]}')
    axs[i].axis('off')
plt.show()


'''ＳＡＬＩＤＡ ＣＯＮＳＯＬＡ:
    Tamaño del conjunto de entrenamiento: 50000
    Tamaño del conjunto de test: 10000
    Tamaño de las imágenes: (32, 32, 3)
    [Nota: Saldrá una ventana nueva con 2 fotos descargadas y su etiqueta]
'''

#█▀█ ▄▀█ █▀ █▀█   ▀█ ▀   █▀▄ █▀▀ █▀▀ █ █▄░█ █ █▀█   ▄▀█ █▀█ █▀█ █░█ █ ▀█▀ █▀▀ █▀▀ ▀█▀ █░█ █▀█ ▄▀█
#█▀▀ █▀█ ▄█ █▄█   █▄ ▄   █▄▀ ██▄ █▀░ █ █░▀█ █ █▀▄   █▀█ █▀▄ ▀▀█ █▄█ █ ░█░ ██▄ █▄▄ ░█░ █▄█ █▀▄ █▀█

def crear_CNN(aumentos=None, normalizacion=False, dropout=0):
    entrada = Input(shape=(32, 32, 3))

    # Aumento de datos
    if aumentos:
        x = aumentos(entrada)
    else:
        x = entrada

    # 1) Bloque extractor de características
    # - Convolución con activación -> Normalización -> Pooling
    x = Conv2D(16, kernel_size=5, padding='same', activation='relu')(x)
    if normalizacion:
        x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=2)(x)
    # - Convolución con activación -> Normalización -> Pooling
    x = Conv2D(32, kernel_size=5, padding='same', activation='relu')(x)
    if normalizacion:
        x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=2)(x)
    # - Dropout
    x = Dropout(dropout)(x)

    # 2) Bloque clasificador de características
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    salida = Dense(10, activation='softmax')(x)

    return Model(inputs=entrada, outputs=salida)

#█▀█ ▄▀█ █▀ █▀█  ▀█ ▀   █▀▀ █▀█ █▀▄▀█ █▀█ █ █░░ ▄▀█ █▀█   █▄█   █▀▀ █▄░█ ▀█▀ █▀█ █▀▀ █▄░█ ▄▀█ █▀█
#█▀▀ █▀█ ▄█ █▄█  ▄█ ▄   █▄▄ █▄█ █░▀░█ █▀▀ █ █▄▄ █▀█ █▀▄   ░█░   ██▄ █░▀█ ░█░ █▀▄ ██▄ █░▀█ █▀█ █▀▄

#█▀▄▀█ █▀█ █▀▄ █▀▀ █░░ █▀█
#█░▀░█ █▄█ █▄▀ ██▄ █▄▄ █▄█

# Caso base
modelo_base = crear_CNN()
modelo_base.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

# Model checkpoint
checkpoint = ModelCheckpoint(filepath='modelo_base.h5',
                             monitor='val_accuracy',
                             save_best_only=True,
                             save_weights_only=True,
                             verbose=1)

# Entrenamos el modelo base
historia_base = modelo_base.fit(X_train, Y_train,
                                validation_split=0.2,
                                epochs=50,
                                batch_size=256,
                                callbacks=[checkpoint])

'''ＳＡＬＩＤＡ ＣＯＮＳＯＬＡ:
    Epoch 1/50
    154/157 [============================>.] - ETA: 0s - loss: 1.7031 - accuracy: 0.3804
    Epoch 1: val_accuracy improved from -inf to 0.48640, saving model to modelo_base.h5
    157/157 [==============================] - 4s 13ms/step - loss: 1.6987 - accuracy: 0.3824 - val_loss: 1.4494 - val_accuracy: 0.4864
    Epoch 2/50
    153/157 [============================>.] - ETA: 0s - loss: 1.3668 - accuracy: 0.5068
    Epoch 2: val_accuracy improved from 0.48640 to 0.54080, saving model to modelo_base.h5
    157/157 [==============================] - 1s 9ms/step - loss: 1.3649 - accuracy: 0.5071 - val_loss: 1.2962 - val_accuracy: 0.5408
    Epoch 3/50
    152/157 [============================>.] - ETA: 0s - loss: 1.2103 - accuracy: 0.5696
    Epoch 3: val_accuracy improved from 0.54080 to 0.58290, saving model to modelo_base.h5
    157/157 [==============================] - 2s 11ms/step - loss: 1.2073 - accuracy: 0.5703 - val_loss: 1.1777 - val_accuracy: 0.5829
    Epoch 4/50
    152/157 [============================>.] - ETA: 0s - loss: 1.1008 - accuracy: 0.6131
    Epoch 4: val_accuracy improved from 0.58290 to 0.58500, saving model to modelo_base.h5
    157/157 [==============================] - 2s 11ms/step - loss: 1.1005 - accuracy: 0.6134 - val_loss: 1.1824 - val_accuracy: 0.5850
    Epoch 5/50
    152/157 [============================>.] - ETA: 0s - loss: 1.0203 - accuracy: 0.6421
    Epoch 5: val_accuracy improved from 0.58500 to 0.62200, saving model to modelo_base.h5
    157/157 [==============================] - 1s 9ms/step - loss: 1.0186 - accuracy: 0.6429 - val_loss: 1.0846 - val_accuracy: 0.6220
    Epoch 6/50
    153/157 [============================>.] - ETA: 0s - loss: 0.9464 - accuracy: 0.6691
    Epoch 6: val_accuracy improved from 0.62200 to 0.64400, saving model to modelo_base.h5
    157/157 [==============================] - 1s 9ms/step - loss: 0.9452 - accuracy: 0.6694 - val_loss: 1.0196 - val_accuracy: 0.6440
    Epoch 7/50
    ...
    Epoch 50/50
    150/157 [===========================>..] - ETA: 0s - loss: 0.0480 - accuracy: 0.9837
    Epoch 50: val_accuracy did not improve from 0.68380
    157/157 [==============================] - 1s 9ms/step - loss: 0.0488 - accuracy: 0.9836 - val_loss: 3.0943 - val_accuracy: 0.6541
'''

###############################CASO AVANZADO ###########################

# Modelo de aumentos de datos
aumentos = Sequential()
aumentos.add(RandomRotation(0.05, fill_mode='nearest', input_shape=(32, 32, 3)))
aumentos.add(RandomFlip('horizontal'))
aumentos.add(RandomBrightness(0.1, value_range=(0, 1)))

# Visualizamos las imágenes aumentadas de un batch de entrenamiento
imagenes_aug = aumentos(imagenes)
fig, axs = plt.subplots(1, 2, figsize=(10, 10))
for i in range(2):
    axs[i].imshow(imagenes_aug[i])
    axs[i].axis('off')
plt.show()

'''ＳＡＬＩＤＡ ＣＯＮＳＯＬＡ:
    [NOTA: Saldrán 2 imagenes manipuladas para el caso avanzado]
'''

# Caso avanzado
modelo_avan = crear_CNN(aumentos=aumentos, normalizacion=True, dropout=0.5)
modelo_avan.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

# Model checkpoint
checkpoint = ModelCheckpoint(filepath='modelo_avan.h5',
                             monitor='val_accuracy',
                             save_best_only=True,
                             save_weights_only=True,
                             verbose=1)

# Early stopping
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)

# Entrenamos el modelo avanzado
historia_avan = modelo_avan.fit(X_train, Y_train,
                                validation_split=0.2,
                                epochs=50,
                                batch_size=256,
                                callbacks=[checkpoint, early_stop])

'''ＳＡＬＩＤＡ ＣＯＮＳＯＬＡ:
    Epoch 1/50
    155/157 [============================>.] - ETA: 0s - loss: 1.7364 - accuracy: 0.3856
    Epoch 1: val_accuracy improved from -inf to 0.20410, saving model to modelo_avan.h5
    157/157 [==============================] - 6s 19ms/step - loss: 1.7348 - accuracy: 0.3859 - val_loss: 2.9060 - val_accuracy: 0.2041
    Epoch 2/50
    157/157 [==============================] - ETA: 0s - loss: 1.4005 - accuracy: 0.4952
    Epoch 2: val_accuracy improved from 0.20410 to 0.39750, saving model to modelo_avan.h5
    157/157 [==============================] - 2s 15ms/step - loss: 1.4005 - accuracy: 0.4952 - val_loss: 1.7619 - val_accuracy: 0.3975
    Epoch 3/50
    154/157 [============================>.] - ETA: 0s - loss: 1.2734 - accuracy: 0.5462
    Epoch 3: val_accuracy improved from 0.39750 to 0.52810, saving model to modelo_avan.h5
    157/157 [==============================] - 2s 15ms/step - loss: 1.2725 - accuracy: 0.5464 - val_loss: 1.3940 - val_accuracy: 0.5281
    Epoch 4/50
    155/157 [============================>.] - ETA: 0s - loss: 1.1784 - accuracy: 0.5799
    Epoch 4: val_accuracy improved from 0.52810 to 0.54970, saving model to modelo_avan.h5
    157/157 [==============================] - 2s 14ms/step - loss: 1.1784 - accuracy: 0.5799 - val_loss: 1.2904 - val_accuracy: 0.5497
    Epoch 5/50
    154/157 [============================>.] - ETA: 0s - loss: 1.1120 - accuracy: 0.6036
    Epoch 5: val_accuracy improved from 0.54970 to 0.56400, saving model to modelo_avan.h5
    157/157 [==============================] - 2s 15ms/step - loss: 1.1113 - accuracy: 0.6039 - val_loss: 1.2416 - val_accuracy: 0.5640
    Epoch 6/50
    157/157 [==============================] - ETA: 0s - loss: 1.0525 - accuracy: 0.6251
    Epoch 6: val_accuracy improved from 0.56400 to 0.62990, saving model to modelo_avan.h5
    157/157 [==============================] - 3s 16ms/step - loss: 1.0525 - accuracy: 0.6251 - val_loss: 1.0415 - val_accuracy: 0.6299
    Epoch 7/50
    ...
    154/157 [============================>.] - ETA: 0s - loss: 0.5534 - accuracy: 0.8021
    Epoch 40: val_accuracy did not improve from 0.76530
    157/157 [==============================] - 2s 14ms/step - loss: 0.5534 - accuracy: 0.8022 - val_loss: 0.7537 - val_accuracy: 0.7481
    Epoch 40: early stopping
'''

#█▀█ ▄▀█ █▀ █▀█   █░█ ▀   █▀█ █▀▀ █▀ █░█ █░░ ▀█▀ ▄▀█ █▀▄ █▀█ █▀   █▄█   █▀█ █▀█ █▀▀ █▀▄ █ █▀▀ █▀▀ █ █▀█ █▄░█
#█▀▀ █▀█ ▄█ █▄█   ▀▀█ ▄   █▀▄ ██▄ ▄█ █▄█ █▄▄ ░█░ █▀█ █▄▀ █▄█ ▄█   ░█░   █▀▀ █▀▄ ██▄ █▄▀ █ █▄▄ █▄▄ █ █▄█ █░▀█

# Calcular la exactitud de test en función de la mejor exactitud de validación
# 1) Cargamos el mejor modelo
modelo_base.load_weights('modelo_base.h5')
modelo_avan.load_weights('modelo_avan.h5')
# 2) Evaluamos el modelo
test_base = modelo_base.evaluate(X_test, Y_test, verbose=0)
test_avan = modelo_avan.evaluate(X_test, Y_test, verbose=0)
print(f'Exactitud de test del modelo base: {test_base[-1]*100:.2f}%')
print(f'Exactitud de test del modelo avanzado: {test_avan[-1]*100:.2f}%')

# Visualizamos la perdida de entrenamiento y la exactitud de validación
# a lo largo de las épocas para ambos modelos
fig, axs = plt.subplots(1, 2, figsize=(15, 5))
res_comp = {'Pérdida de entrenamiento': 'loss', 'Exactitud de validación': 'val_accuracy'}
for i, (k, v) in enumerate(res_comp.items()):
    axs[i].plot(historia_base.history[v], label='Base')
    axs[i].plot(historia_avan.history[v], label='Avanzado')
    axs[i].set_xlabel('Época')
    axs[i].set_ylabel(k)
    axs[i].legend()
plt.show()


'''ＳＡＬＩＤＡ ＣＯＮＳＯＬＡ:
    Exactitud de test del modelo base: 68.04%
    Exactitud de test del modelo avanzado: 75.38%
'''
#Cálculo de épocas necesitado del modelo avanzado
dif_epoca = np.argmax(historia_avan.history['val_accuracy'] > np.max(historia_base.history['val_accuracy']))
print(dif_epoca)