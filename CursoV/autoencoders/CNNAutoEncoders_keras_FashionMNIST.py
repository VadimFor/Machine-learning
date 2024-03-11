'''
 Los autoencoders son un tipo de red neuronal artificial utilizada para el aprendizaje no supervisado de 
 representaciones de datos. Consisten en dos partes principales: el codificador, que transforma los datos 
 de entrada en una representación de menor dimensionalidad, y el decodificador, que intenta reconstruir 
 los datos de entrada a partir de esta representación comprimida.
 
 
 En un autoencoder, el objetivo principal es aprender una representación comprimida de los datos de entrada 
 (codificación) y luego utilizar esta representación para generar una salida que sea una reconstrucción lo
 más fiel posible de los datos de entrada originales (decodificación).

En términos más técnicos, durante el entrenamiento del autoencoder, se optimiza la función de pérdida 
(en este caso, la entropía cruzada binaria) para minimizar la diferencia entre la entrada y la salida 
reconstruida. Esto significa que el modelo está aprendiendo a "reconstruir" las entradas originales a 
partir de su representación interna comprimida. La calidad de esta reconstrucción se evalúa mediante 
la métrica de pérdida elegida.
'''

#█▀█ ▄▀█ █▀ █▀█   ▄█ ▀   █▀█ █▄▄ ▀█▀ █▀▀ █▄░█ █▀▀ █▀█   █▄█   ▀█▀ █▀█ ▄▀█ █▄░█ █▀ █▀▀ █▀█ █▀█ █▀▄▀█ ▄▀█ █▀█
#█▀▀ █▀█ ▄█ █▄█   ░█ ▄   █▄█ █▄█ ░█░ ██▄ █░▀█ ██▄ █▀▄   ░█░   ░█░ █▀▄ █▀█ █░▀█ ▄█ █▀░ █▄█ █▀▄ █░▀░█ █▀█ █▀▄

#█▀▄ ▄▀█ ▀█▀ █▀█ █▀
#█▄▀ █▀█ ░█░ █▄█ ▄█

import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

tf.random.set_seed(1)  # Fijamos la semilla de TF
np.random.seed(1)  # Fijamos la semilla

(train_images, _), (test_images, _) = tf.keras.datasets.fashion_mnist.load_data()

train_images = train_images.astype('float32') / 255.
test_images = test_images.astype('float32') / 255.

train_images = np.reshape(train_images, (len(train_images), 28, 28, 1))
test_images = np.reshape(test_images, (len(test_images), 28, 28, 1))

print('Train_images shape:', train_images.shape)
print('Test_images shape:', test_images.shape)

''' SALIDA CONSOLA
Train_images shape: (60000, 28, 28)
Test_images shape: (10000, 28, 28)
'''

#█▀█ ▄▀█ █▀ █▀█   ▀█ ▀   █▀▄ █▀▀ █▀▀ █ █▄░█ █ █▀█   ▄▀█ █▀█ █▀█ █░█ █ ▀█▀ █▀▀ █▀▀ ▀█▀ █░█ █▀█ ▄▀█
#█▀▀ █▀█ ▄█ █▄█   █▄ ▄   █▄▀ ██▄ █▀░ █ █░▀█ █ █▀▄   █▀█ █▀▄ ▀▀█ █▄█ █ ░█░ ██▄ █▄▄ ░█░ █▄█ █▀▄ █▀█

from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model

# Entrada
input_img = Input(shape=(28, 28, 1))

# Encoder...
x = Conv2D(8, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(4, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# En este punto tenemos la representación intermedia

# Decoder...
x = Conv2D(4, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)

# Capa de salida con 1 convolución
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# Creamos el autoencoder
autoencoder = Model(input_img, decoded)

# Mostramos un resumen de la red
print(autoencoder.summary())

''' SALIDA CONSOLA
    Model: "model_2"
    _________________________________________________________________
    Layer (type)                Output Shape              Param #   
    =================================================================
    input_2 (InputLayer)        [(None, 28, 28, 1)]       0         
                                                                    
    conv2d (Conv2D)             (None, 28, 28, 8)         80        
                                                                    
    max_pooling2d (MaxPooling2D  (None, 14, 14, 8)        0         
    )                                                               
                                                                    
    conv2d_1 (Conv2D)           (None, 14, 14, 4)         292       
                                                                    
    max_pooling2d_1 (MaxPooling  (None, 7, 7, 4)          0         
    2D)                                                             
                                                                    
    conv2d_2 (Conv2D)           (None, 7, 7, 4)           148       
                                                                    
    up_sampling2d (UpSampling2D  (None, 14, 14, 4)        0         
    )                                                               
                                                                    
    conv2d_3 (Conv2D)           (None, 14, 14, 8)         296       
                                                                    
    up_sampling2d_1 (UpSampling  (None, 28, 28, 8)        0         
    2D)                                                             
                                                                    
    conv2d_4 (Conv2D)           (None, 28, 28, 1)         73        
                                                                    
    =================================================================
    Total params: 889
    Trainable params: 889
    Non-trainable params: 0
    _________________________________________________________________
    None
'''

#█▀█ ▄▀█ █▀ █▀█  ▀█ ▀   █▀▀ █▄░█ ▀█▀ █▀█ █▀▀ █▄░█ ▄▀█ █▀█   █▀▄▀█ █▀█ █▀▄ █▀▀ █░░ █▀█
#█▀▀ █▀█ ▄█ █▄█  ▄█ ▄   ██▄ █░▀█ ░█░ █▀▄ ██▄ █░▀█ █▀█ █▀▄   █░▀░█ █▄█ █▄▀ ██▄ █▄▄ █▄█

#compilamos la red
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Entrenamos la red
history = autoencoder.fit(train_images, train_images,
                          validation_data=(test_images, test_images),
                          epochs=15, batch_size=64, verbose=1)

''' SALIDA CONSOLA
    Epoch 1/15
    938/938 [==============================] - 12s 5ms/step - loss: 0.3344 - val_loss: 0.2934
    Epoch 2/15
    938/938 [==============================] - 5s 5ms/step - loss: 0.2873 - val_loss: 0.2868
    Epoch 3/15
    938/938 [==============================] - 5s 5ms/step - loss: 0.2827 - val_loss: 0.2833
    Epoch 4/15
    938/938 [==============================] - 5s 5ms/step - loss: 0.2802 - val_loss: 0.2813
    Epoch 5/15
    938/938 [==============================] - 5s 6ms/step - loss: 0.2785 - val_loss: 0.2798
    Epoch 6/15
    938/938 [==============================] - 4s 5ms/step - loss: 0.2773 - val_loss: 0.2788
    Epoch 7/15
    938/938 [==============================] - 4s 5ms/step - loss: 0.2765 - val_loss: 0.2781
    Epoch 8/15
    938/938 [==============================] - 5s 5ms/step - loss: 0.2758 - val_loss: 0.2776
    Epoch 9/15
    938/938 [==============================] - 5s 5ms/step - loss: 0.2753 - val_loss: 0.2772
    Epoch 10/15
    938/938 [==============================] - 5s 5ms/step - loss: 0.2749 - val_loss: 0.2767
    Epoch 11/15
    938/938 [==============================] - 4s 5ms/step - loss: 0.2745 - val_loss: 0.2768
    Epoch 12/15
    938/938 [==============================] - 4s 5ms/step - loss: 0.2743 - val_loss: 0.2762
    Epoch 13/15
    938/938 [==============================] - 5s 5ms/step - loss: 0.2740 - val_loss: 0.2759
    Epoch 14/15
    938/938 [==============================] - 5s 5ms/step - loss: 0.2737 - val_loss: 0.2756
    Epoch 15/15
    938/938 [==============================] - 5s 5ms/step - loss: 0.2735 - val_loss: 0.2756
'''

#█▀█ ▄▀█ █▀ █▀█   █░█ ▀   █▀█ █▀▀ █▀ █░█ █░░ ▀█▀ ▄▀█ █▀▄ █▀█ █▀   █▄█   █▀█ █▀█ █▀▀ █▀▄ █ █▀▀ █▀▀ █ █▀█ █▄░█
#█▀▀ █▀█ ▄█ █▄█   ▀▀█ ▄   █▀▄ ██▄ ▄█ █▄█ █▄▄ ░█░ █▀█ █▄▀ █▄█ ▄█   ░█░   █▀▀ █▀▄ ██▄ █▄▀ █ █▄▄ █▄▄ █ █▄█ █░▀█


#----------------------------------------------------
def show_autoencoder_result(test_images, prediction):
  n = 10  # how many digits we will display
  plt.figure(figsize=(20, 4))
  for i in range(n):
      # display original
      ax = plt.subplot(2, n, i + 1)
      plt.imshow(test_images[i].reshape(28, 28))
      plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)

      # display reconstruction
      ax = plt.subplot(2, n, i + 1 + n)
      plt.imshow(prediction[i].reshape(28, 28))
      plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
  plt.show()

# -----------------------------
def plot_learning_curves(hist):
  plt.plot(hist.history['loss'])
  plt.plot(hist.history['val_loss'])
  plt.title('Curvas de aprendizaje')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Conjunto de entrenamiento', 'Conjunto de validación'], loc='upper right')
  plt.show()

plot_learning_curves(history)

prediction = autoencoder.predict( test_images )

show_autoencoder_result( test_images, prediction )


#█▀█ █▀▀ █▀█ █▀█ █▀▀ █▀ █▀▀ █▄░█ ▀█▀ ▄▀█ █▀▀ █ █▀█ █▄░█   █▀▀ █ █░░ ▀█▀ █▀█ █▀█
#█▀▄ ██▄ █▀▀ █▀▄ ██▄ ▄█ ██▄ █░▀█ ░█░ █▀█ █▄▄ █ █▄█ █░▀█   █▀░ █ █▄▄ ░█░ █▀▄ █▄█

'''
* A continuación vamos a ver la representación intermedia para uno de los filtros
'''
#----------------------------------------------------
def show_autoencoder_result2(test_images, prediction):
  n = 10  # how many digits we will display
  plt.figure(figsize=(20, 4))
  for i in range(n):
      # display original
      ax = plt.subplot(2, n, i + 1)
      plt.imshow(test_images[i].reshape(28, 28))
      plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)

      # display reconstruction
      ax = plt.subplot(2, n, i + 1 + n)
      plt.imshow(np.array(prediction[i,:,:,1] * 255., dtype=np.uint8), cmap='gray')
      plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
  plt.show()

modelEncoder = Model(input_img, encoded)

predictions = modelEncoder.predict(test_images)

show_autoencoder_result2(test_images, predictions)
