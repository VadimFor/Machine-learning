import os
import cv2 #pip install opencv-python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import tensorflow as tf

#█▀█ ▄▀█ █▀ █▀█   ▄█ ▀   █▀█ █▄▄ ▀█▀ █▀▀ █▄░█ █▀▀ █▀█   █▄█   ▀█▀ █▀█ ▄▀█ █▄░█ █▀ █▀▀ █▀█ █▀█ █▀▄▀█ ▄▀█ █▀█
#█▀▀ █▀█ ▄█ █▄█   ░█ ▄   █▄█ █▄█ ░█░ ██▄ █░▀█ ██▄ █▀▄   ░█░   ░█░ █▀▄ █▀█ █░▀█ ▄█ █▀░ █▄█ █▀▄ █░▀░█ █▀█ █▀▄

#█▀▄ ▄▀█ ▀█▀ █▀█ █▀
#█▄▀ █▀█ ░█░ █▄█ ▄█

tf.random.set_seed(2)  # Fijamos la semilla de TF
np.random.seed(2)  # Fijamos la semilla

#------------------------------------------------------------------------------
def load_dataset():
  X = []
  Y = []
  for fname_x in os.listdir('images'):
    fname_y = fname_x.replace('.png', '_mask.png')
    img_x = cv2.imread(os.path.join('images', fname_x), cv2.IMREAD_COLOR)
    img_y = cv2.imread(os.path.join('gt', fname_y), cv2.IMREAD_GRAYSCALE)
    X.append( cv2.cvtColor(img_x, cv2.COLOR_BGR2RGB) )
    Y.append(img_y)
  X = np.array(X)
  Y = np.array(Y)
  return train_test_split(X, Y, test_size=0.08, random_state=42)

x_train, x_test, y_train, y_test = load_dataset()

print('Datos para entrenamiento:')
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('Datos para evaluación:')
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

'''
Datos para entrenamiento:
x_train shape: (114, 128, 128, 3)
y_train shape: (114, 128, 128)
Datos para evaluación:
x_test shape: (10, 128, 128, 3)
y_test shape: (10, 128, 128)

'''


#----------------------------------------------------
def show_images(images, labels):
  n = 10  # Número de imágenes a mostrar
  plt.figure(figsize=(20, 4))
  for i in range(n):
      # Mostrar la imagen de entrada
      ax = plt.subplot(2, n, i + 1)
      plt.imshow(images[i])
      #plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)

      # Mostrar las etiquetas
      ax = plt.subplot(2, n, i + 1 + n)
      plt.imshow(labels[i].reshape(128, 128))
      plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
  plt.show()


show_images(x_train, y_train)


def prepare_data(x, y):
  y = np.reshape(y, (len(y), 128, 128, 1))
  x = x.astype(np.float32)
  y = y.astype(np.float32)
  x /= 255.
  y /= 255.
  return x, y

x_train, y_train = prepare_data(x_train, y_train)
x_test, y_test = prepare_data(x_test, y_test)


print('Datos para entrenamiento:')
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('\nDatos para evaluación:')
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

'''
Datos para entrenamiento:
x_train shape: (114, 128, 128, 3)
y_train shape: (114, 128, 128, 1)

Datos para evaluación:
x_test shape: (10, 128, 128, 3)
y_test shape: (10, 128, 128, 1)

'''

#█▀█ ▄▀█ █▀ █▀█   ▀█ ▀   █▀▄ █▀▀ █▀▀ █ █▄░█ █ █▀█   ▄▀█ █▀█ █▀█ █░█ █ ▀█▀ █▀▀ █▀▀ ▀█▀ █░█ █▀█ ▄▀█
#█▀▀ █▀█ ▄█ █▄█   █▄ ▄   █▄▀ ██▄ █▀░ █ █░▀█ █ █▀▄   █▀█ █▀▄ ▀▀█ █▄█ █ ░█░ ██▄ █▄▄ ░█░ █▄█ █▀▄ █▀█

from tensorflow.keras.layers import Input, Dense, Conv2D
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D, Dropout
from tensorflow.keras.models import Model


# Entrada
input_img = Input(shape=(128, 128, 3))


# Encoder...
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = Dropout(0.1)(x)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = Dropout(0.1)(x)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = Dropout(0.1)(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# Decoder...
x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = Dropout(0.1)(x)
x = UpSampling2D((2, 2))(x)

x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = Dropout(0.1)(x)
x = UpSampling2D((2, 2))(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = Dropout(0.1)(x)
x = UpSampling2D((2, 2))(x)


# Capa de salida con 1 convolución
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)


# Creamos el autoencoder y lo compilamos
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')


# Mostramos un resumen de la red
print(autoencoder.summary())


'''
Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 128, 128, 3)]     0         
                                                                 
 conv2d (Conv2D)             (None, 128, 128, 32)      896       
                                                                 
 dropout (Dropout)           (None, 128, 128, 32)      0         
                                                                 
 max_pooling2d (MaxPooling2D  (None, 64, 64, 32)       0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 64, 64, 16)        4624      
                                                                 
 dropout_1 (Dropout)         (None, 64, 64, 16)        0         
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 32, 32, 16)       0         
 2D)                                                             
                                                                 
 conv2d_2 (Conv2D)           (None, 32, 32, 8)         1160      
                                                                 
 dropout_2 (Dropout)         (None, 32, 32, 8)         0         
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 16, 16, 8)        0         
 2D)                                                             
                                                                 
 conv2d_3 (Conv2D)           (None, 16, 16, 8)         584       
                                                                 
 dropout_3 (Dropout)         (None, 16, 16, 8)         0         
                                                                 
 up_sampling2d (UpSampling2D  (None, 32, 32, 8)        0         
 )                                                               
                                                                 
 conv2d_4 (Conv2D)           (None, 32, 32, 16)        1168      
                                                                 
 dropout_4 (Dropout)         (None, 32, 32, 16)        0         
                                                                 
 up_sampling2d_1 (UpSampling  (None, 64, 64, 16)       0         
 2D)                                                             
                                                                 
 conv2d_5 (Conv2D)           (None, 64, 64, 32)        4640      
                                                                 
 dropout_5 (Dropout)         (None, 64, 64, 32)        0         
                                                                 
 up_sampling2d_2 (UpSampling  (None, 128, 128, 32)     0         
 2D)                                                             
                                                                 
 conv2d_6 (Conv2D)           (None, 128, 128, 1)       289       
                                                                 
=================================================================
Total params: 13,361
Trainable params: 13,361
Non-trainable params: 0
_________________________________________________________________
None
'''

#█▀█ ▄▀█ █▀ █▀█  ▀█ ▀   █▀▀ █▄░█ ▀█▀ █▀█ █▀▀ █▄░█ ▄▀█ █▀█   █▀▄▀█ █▀█ █▀▄ █▀▀ █░░ █▀█
#█▀▀ █▀█ ▄█ █▄█  ▄█ ▄   ██▄ █░▀█ ░█░ █▀▄ ██▄ █░▀█ █▀█ █▀▄   █░▀░█ █▄█ █▄▀ ██▄ █▄▄ █▄█
# Entrenamos la red
history = autoencoder.fit(x_train, y_train,
                          validation_data=(x_test, y_test),
                          epochs=200, batch_size=64, verbose=1)

'''
Epoch 1/200
2/2 [==============================] - 23s 1s/step - loss: 0.6833 - val_loss: 0.6614
Epoch 2/200
2/2 [==============================] - 0s 77ms/step - loss: 0.6348 - val_loss: 0.5959
Epoch 3/200
2/2 [==============================] - 0s 78ms/step - loss: 0.5349 - val_loss: 0.4798
Epoch 4/200
2/2 [==============================] - 0s 78ms/step - loss: 0.3799 - val_loss: 0.3156
Epoch 5/200

Epoch 12/200
2/2 [==============================] - 0s 84ms/step - loss: 0.1100 - val_loss: 0.1285
Epoch 13/200
2/2 [==============================] - 0s 77ms/step - loss: 0.0985 - val_loss: 0.1133
Epoch 14/200
2/2 [==============================] - 0s 83ms/step - loss: 0.0852 - val_loss: 0.0983

Epoch 24/200
2/2 [==============================] - 0s 77ms/step - loss: 0.0543 - val_loss: 0.0740
Epoch 25/200
2/2 [==============================] - 0s 77ms/step - loss: 0.0544 - val_loss: 0.0729
Epoch 26/200
2/2 [==============================] - 0s 84ms/step - loss: 0.0532 - val_loss: 0.0719

Epoch 124/200
2/2 [==============================] - 0s 100ms/step - loss: 0.0221 - val_loss: 0.0265
Epoch 125/200

Epoch 132/200
2/2 [==============================] - 0s 86ms/step - loss: 0.0206 - val_loss: 0.0253
Epoch 133/200
2/2 [==============================] - 0s 84ms/step - loss: 0.0200 - val_loss: 0.0254
Epoch 134/200
2/2 [==============================] - 0s 73ms/step - loss: 0.0207 - val_loss: 0.0251
Epoch 135/200
2/2 [==============================] - 0s 88ms/step - loss: 0.0203 - val_loss: 0.0253
Epoch 136/200

2/2 [==============================] - 0s 82ms/step - loss: 0.0194 - val_loss: 0.0240
Epoch 142/200
2/2 [==============================] - 0s 82ms/step - loss: 0.0196 - val_loss: 0.0236
Epoch 143/200
2/2 [==============================] - 0s 84ms/step - loss: 0.0185 - val_loss: 0.0244
Epoch 144/200

Epoch 152/200
2/2 [==============================] - 0s 82ms/step - loss: 0.0176 - val_loss: 0.0229
Epoch 153/200
2/2 [==============================] - 0s 82ms/step - loss: 0.0174 - val_loss: 0.0228
Epoch 154/200
2/2 [==============================] - 0s 77ms/step - loss: 0.0169 - val_loss: 0.0226
Epoch 155/200

2/2 [==============================] - 0s 77ms/step - loss: 0.0161 - val_loss: 0.0221
Epoch 162/200
2/2 [==============================] - 0s 81ms/step - loss: 0.0160 - val_loss: 0.0216
Epoch 163/200
2/2 [==============================] - 0s 87ms/step - loss: 0.0165 - val_loss: 0.0217
Epoch 164/200

Epoch 169/200
2/2 [==============================] - 0s 85ms/step - loss: 0.0162 - val_loss: 0.0215
Epoch 170/200
2/2 [==============================] - 0s 83ms/step - loss: 0.0163 - val_loss: 0.0206
Epoch 171/200
2/2 [==============================] - 0s 86ms/step - loss: 0.0157 - val_loss: 0.0207

Epoch 179/200
2/2 [==============================] - 0s 82ms/step - loss: 0.0156 - val_loss: 0.0208

2/2 [==============================] - 0s 82ms/step - loss: 0.0147 - val_loss: 0.0201
Epoch 186/200
2/2 [==============================] - 0s 82ms/step - loss: 0.0144 - val_loss: 0.0194
Epoch 187/200
2/2 [==============================] - 0s 82ms/step - loss: 0.0145 - val_loss: 0.0195
Epoch 188/200
2/2 [==============================] - 0s 84ms/step - loss: 0.0146 - val_loss: 0.0195

Epoch 195/200
2/2 [==============================] - 0s 95ms/step - loss: 0.0141 - val_loss: 0.0187
Epoch 196/200
2/2 [==============================] - 0s 102ms/step - loss: 0.0136 - val_loss: 0.0186
Epoch 197/200
2/2 [==============================] - 0s 98ms/step - loss: 0.0136 - val_loss: 0.0179
Epoch 198/200
2/2 [==============================] - 0s 97ms/step - loss: 0.0136 - val_loss: 0.0185
Epoch 199/200
2/2 [==============================] - 0s 90ms/step - loss: 0.0136 - val_loss: 0.0182
Epoch 200/200
2/2 [==============================] - 0s 85ms/step - loss: 0.0137 - val_loss: 0.0194
'''

#█▀█ ▄▀█ █▀ █▀█   █░█ ▀   █▀█ █▀▀ █▀ █░█ █░░ ▀█▀ ▄▀█ █▀▄ █▀█ █▀   █▄█   █▀█ █▀█ █▀▀ █▀▄ █ █▀▀ █▀▀ █ █▀█ █▄░█
#█▀▀ █▀█ ▄█ █▄█   ▀▀█ ▄   █▀▄ ██▄ ▄█ █▄█ █▄▄ ░█░ █▀█ █▄▀ █▄█ ▄█   ░█░   █▀▀ █▀▄ ██▄ █▄▀ █ █▄▄ █▄▄ █ █▄█ █░▀█

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


prediction = autoencoder.predict( x_test )

prediction = (prediction > 0.2).astype('uint8')

prediction *= 255

show_images( x_test, prediction )

test_flat = (y_test > 0.2).astype('uint8')
test_flat = test_flat.flatten()
pred_flat = (prediction / 255).flatten()


from sklearn.metrics import f1_score

print( 'Acierto F1: {:.4f}'.format( f1_score(test_flat, pred_flat, average='macro') ) )