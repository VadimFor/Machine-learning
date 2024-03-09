import os
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.mobilenet import preprocess_input
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose,\
    BatchNormalization, Dropout, Concatenate
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

#█▀█ ▄▀█ █▀ █▀█   ▄█ ▀   █▀█ █▄▄ ▀█▀ █▀▀ █▄░█ █▀▀ █▀█   █▄█   ▀█▀ █▀█ ▄▀█ █▄░█ █▀ █▀▀ █▀█ █▀█ █▀▄▀█ ▄▀█ █▀█
#█▀▀ █▀█ ▄█ █▄█   ░█ ▄   █▄█ █▄█ ░█░ ██▄ █░▀█ ██▄ █▀▄   ░█░   ░█░ █▀▄ █▀█ █░▀█ ▄█ █▀░ █▄█ █▀▄ █░▀░█ █▀█ █▀▄

#█▀▄ ▄▀█ ▀█▀ █▀█ █▀
#█▄▀ █▀█ ░█░ █▄█ ▄█

''' descargar datos
    wget -q http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
    wget -q http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
    tar -xf images.tar.gz
    tar -xf annotations.tar.gz
'''
# Rutas a los directorios de imágenes y máscaras
imgs_dir = 'images'
mascs_dir = 'annotations/trimaps/'

# Tamaño de las imágenes y máscaras
tam_img = (160, 160)

# Número de imágenes a cargar
num_imgs = 3000

def cargar_datos(imgs_dir, mascs_dir, tam_img, num_imgs=None):
    X = []
    Y = []
    img_list = sorted([i for i in os.listdir(imgs_dir)\
                       if i.endswith('.jpg') and not i.startswith('.')])

    if num_imgs is not None:
        img_list = img_list[:num_imgs]

    for inom in img_list:
        # Imagen
        img = img_to_array(load_img(os.path.join(imgs_dir, inom),\
                                    target_size=tam_img))
        # Máscara
        mnom = inom.replace('.jpg', '.png')
        masc = img_to_array(load_img(os.path.join(mascs_dir, mnom),\
                                     color_mode='grayscale',\
                                     target_size=tam_img))
        # Restamos 1 a todos los valores de la máscara
        masc -= 1

        X.append(img)
        Y.append(masc)

    X = preprocess_input(np.array(X))   # Normalizamos
    Y = np.array(Y, dtype='uint8')      # Convertimos a enteros

    return X, Y

X, Y = cargar_datos(imgs_dir, mascs_dir, tam_img, num_imgs)
print('¡Imágenes cargadas!')
print('Rango de valores de X: [{}, {}]'.format(X.min(), X.max()))

'''ＳＡＬＩＤＡ ＣＯＮＳＯＬＡ:
    ¡Imágenes cargadas!
    Rango de valores de X: [-1.0, 1.0]
'''

# Tamaño de los datos
print(f'Tamaño de las imágenes: {X.shape[1:]}')
print(f'Tamaño de las máscaras: {Y.shape[1:]}')

# Visualizamos las imágenes y las máscaras
fig, axs = plt.subplots(2, 5, figsize=(15, 6))
for i in range(5):
    # Trasladamos los valores de la imagen de [-1, 1]
    # a [0, 1] para su visualización
    axs[0, i].imshow((X[i] + 1) / 2)
    axs[0, i].axis('off')
    axs[1, i].imshow(Y[i], cmap='gray')
    axs[1, i].axis('off')
axs[0, 2].set_title('Imágenes')
axs[1, 2].set_title('Máscaras')
plt.show()

'''ＳＡＬＩＤＡ ＣＯＮＳＯＬＡ:
    Tamaño de las imágenes: (160, 160, 3)
    Tamaño de las máscaras: (160, 160, 1)
    [Nota: Saldrán una vetnaa nueva con imagenes de animales y la máscara aplicada.]
'''


X_train, X_test, Y_train, Y_test = train_test_split(X, Y,\
                                                    test_size=0.2,\
                                                    random_state=42)
print('¡Particiones realizadas!')
print(f'Tamaño del conjunto de entrenamiento: {X_train.shape[0]}')
print(f'Tamaño del conjunto de test: {X_test.shape[0]}')

'''ＳＡＬＩＤＡ ＣＯＮＳＯＬＡ:
    ¡Particiones realizadas!
    Tamaño del conjunto de entrenamiento: 2400
    Tamaño del conjunto de test: 600
'''


#█▀█ ▄▀█ █▀ █▀█   ▀█ ▀   █▀▄ █▀▀ █▀▀ █ █▄░█ █ █▀█   ▄▀█ █▀█ █▀█ █░█ █ ▀█▀ █▀▀ █▀▀ ▀█▀ █░█ █▀█ ▄▀█
#█▀▀ █▀█ ▄█ █▄█   █▄ ▄   █▄▀ ██▄ █▀░ █ █░▀█ █ █▀▄   █▀█ █▀▄ ▀▀█ █▄█ █ ░█░ ██▄ █▄▄ ░█░ █▄█ █▀▄ █▀█

def crear_modelo(tam_entrada=(160, 160, 3), num_clases=3):
    # Cargamos el modelo preentrenado MobileNetV2 con ImageNet
    # y le indicamos que no incluya las capas de salida
    modelo_base = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=tam_entrada
    )

    # Codificador (base)
    # Obtenemos la salida de la capa 'block_6_expand_relu'
    # del modelo base
    # Tamaño salida: 20x20x192
    x = modelo_base.get_layer('block_6_expand_relu').output

    # Decodificador (personalizado)
    # Upsampling 1
    # Tamaño de salida: 40x40x512
    x = Conv2DTranspose(512, 3, 2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Upsampling 2 + residual
    # Tamaño de salida: 80x80x512
    x = Concatenate()(
        [x, modelo_base.get_layer('block_3_expand_relu').output])
    x = Conv2DTranspose(512, 3, 2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Upsampling 3 + residual
    # Tamaño de salida: 160x160x256
    x = Concatenate()(
        [x, modelo_base.get_layer('block_1_expand_relu').output])
    x = Conv2DTranspose(256, 3, 2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Capa de salida con convolución 1x1
    salida = Conv2D(num_clases, 1, padding='same',\
                    activation='softmax')(x)

    # Creamos el modelo final
    modelo = Model(inputs=modelo_base.input, outputs=salida)

    return modelo

modelo = crear_modelo()
print(modelo.summary())

'''ＳＡＬＩＤＡ ＣＯＮＳＯＬＡ:
    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160_no_top.h5
    9406464/9406464 [==============================] - 2s 0us/step
    Model: "model"
    __________________________________________________________________________________________________
    Layer (type)                   Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_1 (InputLayer)           [(None, 160, 160, 3  0           []                               
                                    )]                                                                
                                                                                                    
    Conv1 (Conv2D)                 (None, 80, 80, 32)   864         ['input_1[0][0]']                
                                                                                                    
    bn_Conv1 (BatchNormalization)  (None, 80, 80, 32)   128         ['Conv1[0][0]']                  
                                                                                                    
    Conv1_relu (ReLU)              (None, 80, 80, 32)   0           ['bn_Conv1[0][0]']               
                                                                                                    
    expanded_conv_depthwise (Depth  (None, 80, 80, 32)  288         ['Conv1_relu[0][0]']             
    wiseConv2D)                                                                                      
                                                                                                    
    expanded_conv_depthwise_BN (Ba  (None, 80, 80, 32)  128         ['expanded_conv_depthwise[0][0]']
    tchNormalization)                                                                                
                                                                                                    
    expanded_conv_depthwise_relu (  (None, 80, 80, 32)  0           ['expanded_conv_depthwise_BN[0][0
    ReLU)                                                           ]']                              
                                                                                                    
    expanded_conv_project (Conv2D)  (None, 80, 80, 16)  512         ['expanded_conv_depthwise_relu[0]
    ...
    Trainable params: 5,375,043
    Non-trainable params: 6,464
    __________________________________________________________________________________________________
    None
'''

#█▀█ ▄▀█ █▀ █▀█  ▀█ ▀   █▀▀ █▀█ █▀▄▀█ █▀█ █ █░░ ▄▀█ █▀█   █▄█   █▀▀ █▄░█ ▀█▀ █▀█ █▀▀ █▄░█ ▄▀█ █▀█
#█▀▀ █▀█ ▄█ █▄█  ▄█ ▄   █▄▄ █▄█ █░▀░█ █▀▀ █ █▄▄ █▀█ █▀▄   ░█░   ██▄ █░▀█ ░█░ █▀▄ ██▄ █░▀█ █▀█ █▀▄

#█▀▄▀█ █▀█ █▀▄ █▀▀ █░░ █▀█
#█░▀░█ █▄█ █▄▀ ██▄ █▄▄ █▄█

################ENTRENAMIENTO SOLO DE LA CAPAS NUEVAS AÑADIDAS AL MODELO DESCARGADO
# Congelamos las capas del modelo base
for capa in modelo.layers:
    capa.trainable = False
    if capa.name == 'block_6_expand_relu':
        break

# Compilamos el modelo
modelo.compile(optimizer='adam',\
               loss='sparse_categorical_crossentropy',\
               metrics=['accuracy'])

# Entrenamos el modelo
historial1 = modelo.fit(X_train, Y_train,\
                        validation_data=(X_test, Y_test),\
                        epochs=10, batch_size=32, verbose=1)


'''ＳＡＬＩＤＡ ＣＯＮＳＯＬＡ:
    Epoch 1/10
    75/75 [==============================] - 130s 1s/step - loss: 0.8615 - accuracy: 0.7004 - val_loss: 0.8590 - val_accuracy: 0.6759
    Epoch 2/10
    75/75 [==============================] - 99s 1s/step - loss: 0.5967 - accuracy: 0.7629 - val_loss: 0.6198 - val_accuracy: 0.7530
    Epoch 3/10
    75/75 [==============================] - 99s 1s/step - loss: 0.5562 - accuracy: 0.7762 - val_loss: 0.5293 - val_accuracy: 0.7887
    Epoch 4/10
    75/75 [==============================] - 99s 1s/step - loss: 0.5406 - accuracy: 0.7816 - val_loss: 0.5148 - val_accuracy: 0.7941
    Epoch 5/10
    75/75 [==============================] - 99s 1s/step - loss: 0.5323 - accuracy: 0.7849 - val_loss: 0.5111 - val_accuracy: 0.7942
    Epoch 6/10
    75/75 [==============================] - 99s 1s/step - loss: 0.5285 - accuracy: 0.7858 - val_loss: 0.5131 - val_accuracy: 0.7915
    Epoch 7/10
    75/75 [==============================] - 99s 1s/step - loss: 0.5235 - accuracy: 0.7876 - val_loss: 0.5016 - val_accuracy: 0.7966
    Epoch 8/10
    75/75 [==============================] - 99s 1s/step - loss: 0.5222 - accuracy: 0.7882 - val_loss: 0.5092 - val_accuracy: 0.7936
    Epoch 9/10
    75/75 [==============================] - 99s 1s/step - loss: 0.5215 - accuracy: 0.7883 - val_loss: 0.5020 - val_accuracy: 0.7966
    Epoch 10/10
    75/75 [==============================] - 99s 1s/step - loss: 0.5184 - accuracy: 0.7894 - val_loss: 0.5012 - val_accuracy: 0.7972
'''

#####################ENTRENAMIENTO DEL MODELO COMPLETO

# Descongelamos las capas del modelo base
for capa in modelo.layers:
    capa.trainable = True
    if capa.name == 'block_6_expand_relu':
        break

# Compilamos el modelo usando un learning rate muy bajo
modelo.compile(optimizer=Adam(1e-5),\
               loss='sparse_categorical_crossentropy',\
               metrics=['accuracy'])

# Entrenamos el modelo
historial2 = modelo.fit(X_train, Y_train,\
                        validation_data=(X_test, Y_test),\
                        epochs=20, batch_size=32, verbose=1)

'''ＳＡＬＩＤＡ ＣＯＮＳＯＬＡ:
    Epoch 1/20
    75/75 [==============================] - 117s 1s/step - loss: 0.5151 - accuracy: 0.7910 - val_loss: 0.5130 - val_accuracy: 0.7902
    Epoch 2/20
    75/75 [==============================] - 103s 1s/step - loss: 0.5067 - accuracy: 0.7949 - val_loss: 0.5202 - val_accuracy: 0.7862
    Epoch 3/20
    75/75 [==============================] - 103s 1s/step - loss: 0.5021 - accuracy: 0.7971 - val_loss: 0.5146 - val_accuracy: 0.7889
    Epoch 4/20
    75/75 [==============================] - 103s 1s/step - loss: 0.4960 - accuracy: 0.8000 - val_loss: 0.5068 - val_accuracy: 0.7928
    Epoch 5/20
    75/75 [==============================] - 103s 1s/step - loss: 0.4917 - accuracy: 0.8021 - val_loss: 0.4975 - val_accuracy: 0.7972
    Epoch 6/20
    ...
    Epoch 19/20
    75/75 [==============================] - 100s 1s/step - loss: 0.4465 - accuracy: 0.8226 - val_loss: 0.4332 - val_accuracy:Epoch 19/20
    75/75 [==============================] - 100s 1s/step - loss: 0.4465 - accuracy: 0.8226 - val_loss: 0.4332 - val_accuracy: 0.8277
    Epoch 20/20
    75/75 [==============================] - 100s 1s/step - loss: 0.4446 - accuracy: 0.8234 - val_loss: 0.4315 - val_accuracy: 0.8284
    Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings... 0.8277
    Epoch 20/20
    75/75 [==============================] - 100s 1s/step - loss: 0.4446 - accuracy: 0.8234 - val_loss: 0.4315 - val_accuracy: 0.8284

'''


#█▀█ ▄▀█ █▀ █▀█   █░█ ▀   █▀█ █▀▀ █▀ █░█ █░░ ▀█▀ ▄▀█ █▀▄ █▀█ █▀   █▄█   █▀█ █▀█ █▀▀ █▀▄ █ █▀▀ █▀▀ █ █▀█ █▄░█
#█▀▀ █▀█ ▄█ █▄█   ▀▀█ ▄   █▀▄ ██▄ ▄█ █▄█ █▄▄ ░█░ █▀█ █▄▀ █▄█ ▄█   ░█░   █▀▀ █▀▄ ██▄ █▄▀ █ █▄▄ █▄▄ █ █▄█ █░▀█


# Concatemanos los historiales de entrenamiento
historial = historial1.history
for k in historial2.history:
    historial[k] += historial2.history[k]

fig, axs = plt.subplots(1, 2, figsize=(15, 5))
# Pérdida
axs[0].plot(historial['loss'], label='Entrenamiento')
axs[0].plot(historial['val_loss'], label='Validación')
axs[0].set_xlabel('Época')
axs[0].set_ylabel('Pérdida')
axs[0].legend()
# Exactitud
axs[1].plot(historial['accuracy'], label='Entrenamiento')
axs[1].plot(historial['val_accuracy'], label='Validación')
axs[1].set_xlabel('Época')
axs[1].set_ylabel('Exactitud (%)')
axs[1].legend()
plt.show()

'''ＳＡＬＩＤＡ ＣＯＮＳＯＬＡ:
    [Nota: Saldrá ventana nueva con 2 gráficas]
'''

###################### PREDICCION

# Obtenemos las predicciones
Y_pred = modelo.predict(X_test, verbose=0)
Y_pred = Y_pred.argmax(axis=-1)

# Visualizamos las predicciones
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
for i in range(2):
    ax[i, 0].set_title('Imagen de test')
    ax[i, 0].imshow((X_test[i] + 1) / 2)
    ax[i, 1].set_title('Máscara de test')
    ax[i, 1].imshow(Y_test[i], cmap='gray')
    ax[i, 2].set_title('Predicción')
    ax[i, 2].imshow(Y_pred[i], cmap='gray')
    # Hacemos los ejes invisibles
    for j in range(3):
        ax[i, j].set_axis_off()
plt.show()

'''ＳＡＬＩＤＡ ＣＯＮＳＯＬＡ:
    [Nota: Saldrá ventana nueva con imagenes y la prediccion de las mascara]
'''