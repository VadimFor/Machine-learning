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

# Cargamos los datos
(train_images, _), (test_images, _) = tf.keras.datasets.fashion_mnist.load_data()


# Mostramos 15 imágenes aleatorias de este dataset
n = 15
index = np.random.randint(len(train_images), size=n)
plt.figure(figsize=(n*1.5, 1.5))
for i in np.arange(n):
    ax = plt.subplot(1,n,i+1)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.imshow(train_images[index[i]], cmap='gray')
plt.show()


print('Train_images shape:', train_images.shape)
print('Test_images shape:', test_images.shape)

''' SALIDA CONSOLA
Train_images shape: (60000, 28, 28)
Test_images shape: (10000, 28, 28)
'''

#--------------------------------------TRANSFORMAR
TRAINING_SIZE = len(train_images)
TEST_SIZE = len(test_images)

# Convertir de (N, 28, 28) a (N, 784)
train_images = np.reshape(train_images, (TRAINING_SIZE, 784))
test_images = np.reshape(test_images, (TEST_SIZE, 784))


# Convertir un array de unit8 (enteros) en float32 (decimales)
train_images = train_images.astype(np.float32)
test_images = test_images.astype(np.float32)

# Normalizar los datos al rango [0,1]
# Como los valores de gris de las imágenes están en el rango [0,255]
# simplemente dividimos por 255
train_images /= 255
test_images /= 255


#█▀█ ▄▀█ █▀ █▀█   ▀█ ▀   █▀▄ █▀▀ █▀▀ █ █▄░█ █ █▀█   ▄▀█ █▀█ █▀█ █░█ █ ▀█▀ █▀▀ █▀▀ ▀█▀ █░█ █▀█ ▄▀█
#█▀▀ █▀█ ▄█ █▄█   █▄ ▄   █▄▀ ██▄ █▀░ █ █░▀█ █ █▀▄   █▀█ █▀▄ ▀▀█ █▄█ █ ░█░ ██▄ █▄▄ ░█░ █▄█ █▀▄ █▀█

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense


# Tamaño de la representación intermedia
encoding_dim = 32

# Entrada
input_img = Input(shape=(784,))

# Codificación intermedia con 32 neuronas
encoded = Dense(encoding_dim, activation='relu')(input_img)

# Capa de salida con 784 neuronas
decoded = Dense(784, activation='sigmoid')(encoded)

# Creamos el modelo usando la API funcional de Keras
model = Model(input_img, decoded)


# Mostramos un resumen de la red
print(model.summary())

''' SALIDA CONSOLA
    Model: "model"
    _________________________________________________________________
    Layer (type)                Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)        [(None, 784)]             0         
                                                                    
    dense (Dense)               (None, 32)                25120     
                                                                    
    dense_1 (Dense)             (None, 784)               25872     
                                                                    
    =================================================================
    Total params: 50,992
    Trainable params: 50,992
    Non-trainable params: 0
    _________________________________________________________________
    None
'''

#█▀█ ▄▀█ █▀ █▀█  ▀█ ▀   █▀▀ █▄░█ ▀█▀ █▀█ █▀▀ █▄░█ ▄▀█ █▀█   █▀▄▀█ █▀█ █▀▄ █▀▀ █░░ █▀█
#█▀▀ █▀█ ▄█ █▄█  ▄█ ▄   ██▄ █░▀█ ░█░ █▀▄ ██▄ █░▀█ █▀█ █▀▄   █░▀░█ █▄█ █▄▀ ██▄ █▄▄ █▄█

# La compilamos
model.compile(optimizer='adam', loss='binary_crossentropy')

# Iniciamos el entrenamiento
model.fit(train_images, train_images, epochs=15, batch_size=64, verbose=1)

''' SALIDA CONSOLA
    Epoch 1/15
    938/938 [==============================] - 8s 3ms/step - loss: 0.3476
    Epoch 2/15
    938/938 [==============================] - 3s 3ms/step - loss: 0.2969
    Epoch 3/15
    938/938 [==============================] - 3s 3ms/step - loss: 0.2890
    Epoch 4/15
    938/938 [==============================] - 3s 3ms/step - loss: 0.2864
    Epoch 5/15
    938/938 [==============================] - 3s 3ms/step - loss: 0.2852
    Epoch 6/15
    938/938 [==============================] - 3s 3ms/step - loss: 0.2846
    Epoch 7/15
    938/938 [==============================] - 3s 3ms/step - loss: 0.2841
    Epoch 8/15
    938/938 [==============================] - 3s 3ms/step - loss: 0.2838
    Epoch 9/15
    938/938 [==============================] - 3s 3ms/step - loss: 0.2835
    Epoch 10/15
    938/938 [==============================] - 3s 3ms/step - loss: 0.2833
    Epoch 11/15
    938/938 [==============================] - 3s 3ms/step - loss: 0.2832
    Epoch 12/15
    938/938 [==============================] - 3s 4ms/step - loss: 0.2830
    Epoch 13/15
    938/938 [==============================] - 3s 3ms/step - loss: 0.2829
    Epoch 14/15
    938/938 [==============================] - 3s 3ms/step - loss: 0.2828
    Epoch 15/15
    938/938 [==============================] - 3s 3ms/step - loss: 0.2827
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

# Utilizamos el modelo aprendido para predecir el resultado del conjunto de test
prediction = model.predict( test_images )

show_autoencoder_result( test_images, prediction )


print('Vamos a mostrar cómo codifica las imágenes')
modelEncoder = Model(input_img, encoded)

predictions = modelEncoder.predict(test_images)

print('\nVector de características de la imagen 150:')
print(test_images[150])
print('\nVector codificado:')
print(predictions[150])

''' SALIDA CONSOLA
    [Nota: Saldrán 2 filas de fotos, la fila de arriba son las fotos orignales de entrada 
    mientras que la fila de abajo son las fotos que ha predicho el modelo en relacion
    a la foto de arriba.]
'''
