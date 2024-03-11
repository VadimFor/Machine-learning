import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist


#█▀█ ▄▀█ █▀ █▀█   ▄█ ▀   █▀█ █▄▄ ▀█▀ █▀▀ █▄░█ █▀▀ █▀█   █▄█   ▀█▀ █▀█ ▄▀█ █▄░█ █▀ █▀▀ █▀█ █▀█ █▀▄▀█ ▄▀█ █▀█
#█▀▀ █▀█ ▄█ █▄█   ░█ ▄   █▄█ █▄█ ░█░ ██▄ █░▀█ ██▄ █▀▄   ░█░   ░█░ █▀▄ █▀█ █░▀█ ▄█ █▀░ █▄█ █▀▄ █░▀░█ █▀█ █▀▄

#█▀▄ ▄▀█ ▀█▀ █▀█ █▀
#█▄▀ █▀█ ░█░ █▄█ ▄█
tf.random.set_seed(2)  # Fijamos la semilla de TF
np.random.seed(2)  # Fijamos la semilla

# Descargamos la base de datos
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

labels = ['Camiseta/Top', 'Pantalón', 'Suéter', 'Vestido', 'Abrigo', 'Sandalia', 'Camisa', 'Zapatilla', 'Bolso/a', 'Botín']


# Mostramos algunas imágenes
n = 15
index = np.random.randint(len(x_train), size=n)
plt.figure(figsize=(n*1.5, 1.5))
for i in np.arange(n):
    ax = plt.subplot(1,n,i+1)
    ax.set_title( labels[ y_train[ index[i] ] ] )
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.imshow(x_train[index[i]], cmap='gray')
plt.show()


# Mostramos la forma de los datos
print('Datos para entrenamiento:')
print(' - x_train: {}'.format( x_train.shape ))
print(' - y_train: {}'.format( y_train.shape ))
print('Datos para evaluación:')
print(' - x_test: {}'.format( x_test.shape ))
print(' - y_test: {}'.format( y_test.shape ))


# --------------------- TRANSFORMAR DATOS
def prepare_data(x):
  x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)    # Redimensionamos para añadir el canal
  x = x.astype(np.float32)   # Transformamos a decimal
  x /= 255.0                 # Normalizamos entre 0 y 1
  return x

x_train = prepare_data(x_train)
x_test  = prepare_data(x_test)


# Transformamos las etiquetas a categórico (one-hot)
NUM_LABELS = 10
y_train  = tf.keras.utils.to_categorical(y_train, NUM_LABELS)
y_test = tf.keras.utils.to_categorical(y_test, NUM_LABELS)


# Para los primeros ejemplos vamos a limitar el número de imágenes de
# entrenamiento a 50. Además nos guardamos un backup con todas las imágenes.
x_train_backup = x_train.copy()
y_train_backup = y_train.copy()
x_train = x_train[:50]
y_train = y_train[:50]


# Mostramos (de nuevo) las dimensiones de los datos
print('Datos para entrenamiento:')
print(' - x_train: {}'.format( x_train.shape ))
print(' - y_train: {}'.format( y_train.shape ))
print('Datos para evaluación:')
print(' - x_test: {}'.format( x_test.shape ))
print(' - y_test: {}'.format( y_test.shape ))

#█▀█ ▄▀█ █▀ █▀█   ▀█ ▀   █▀▄ █▀▀ █▀▀ █ █▄░█ █ █▀█   ▄▀█ █▀█ █▀█ █░█ █ ▀█▀ █▀▀ █▀▀ ▀█▀ █░█ █▀█ ▄▀█
#█▀▀ █▀█ ▄█ █▄█   █▄ ▄   █▄▀ ██▄ █▀░ █ █░▀█ █ █▀▄   █▀█ █▀▄ ▀▀█ █▄█ █ ░█░ ██▄ █▄▄ ░█░ █▄█ █▀▄ █▀█


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

model1 = Sequential()

# Capa convolucional con 1 filtro de tamaño 3x3 seguida de un MaxPooling de 2x2
model1.add(Conv2D(1, (3, 3), activation='relu', input_shape=x_train.shape[1:]))
model1.add(MaxPooling2D(pool_size=(2, 2)))

# Capa Fully Connected
model1.add(Flatten())
model1.add(Dense(NUM_LABELS, activation='softmax'))

print(model1.summary())

''' SALIDA CONSOLA

    Model: "sequential_7"
    _________________________________________________________________
    Layer (type)                Output Shape              Param #   
    =================================================================
    conv2d_12 (Conv2D)          (None, 26, 26, 1)         10        
                                                                    
    max_pooling2d_12 (MaxPoolin  (None, 13, 13, 1)        0         
    g2D)                                                            
                                                                    
    flatten_7 (Flatten)         (None, 169)               0         
                                                                    
    dense_7 (Dense)             (None, 10)                1700      
                                                                    
    =================================================================
    Total params: 1,710
    Trainable params: 1,710
    Non-trainable params: 0
    _________________________________________________________________
    None
'''

#█▀█ ▄▀█ █▀ █▀█  ▀█ ▀   █▀▀ █▄░█ ▀█▀ █▀█ █▀▀ █▄░█ ▄▀█ █▀█   █▀▄▀█ █▀█ █▀▄ █▀▀ █░░ █▀█
#█▀▀ █▀█ ▄█ █▄█  ▄█ ▄   ██▄ █░▀█ ░█░ █▀▄ ██▄ █░▀█ █▀█ █▀▄   █░▀░█ █▄█ █▄▀ ██▄ █▄▄ █▄█

# Compilamos la red
model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'] )

# Entrenamos durante 10 épocas con un batch de 32
history = model1.fit(x_train, y_train,
                     validation_data=(x_test, y_test),
                     batch_size=32, epochs=10, verbose=1)

''' SALIDA CONSOLA

    Epoch 1/10
    2/2 [==============================] - 2s 1s/step - loss: 2.2786 - accuracy: 0.0800 - val_loss: 2.2701 - val_accuracy: 0.1230
    Epoch 2/10
    2/2 [==============================] - 1s 912ms/step - loss: 2.2631 - accuracy: 0.1200 - val_loss: 2.2617 - val_accuracy: 0.1360
    Epoch 3/10
    2/2 [==============================] - 1s 1s/step - loss: 2.2492 - accuracy: 0.1600 - val_loss: 2.2535 - val_accuracy: 0.1495
    Epoch 4/10
    2/2 [==============================] - 1s 649ms/step - loss: 2.2349 - accuracy: 0.1600 - val_loss: 2.2454 - val_accuracy: 0.1615
    Epoch 5/10
    2/2 [==============================] - 1s 649ms/step - loss: 2.2212 - accuracy: 0.1600 - val_loss: 2.2371 - val_accuracy: 0.1746
    Epoch 6/10
    2/2 [==============================] - 1s 659ms/step - loss: 2.2070 - accuracy: 0.2200 - val_loss: 2.2287 - val_accuracy: 0.1909
    Epoch 7/10
    2/2 [==============================] - 1s 631ms/step - loss: 2.1924 - accuracy: 0.3000 - val_loss: 2.2203 - val_accuracy: 0.2063
    Epoch 8/10
    2/2 [==============================] - 1s 651ms/step - loss: 2.1785 - accuracy: 0.3000 - val_loss: 2.2116 - val_accuracy: 0.2194
    Epoch 9/10
    2/2 [==============================] - 1s 621ms/step - loss: 2.1636 - accuracy: 0.3000 - val_loss: 2.2028 - val_accuracy: 0.2340
    Epoch 10/10
    2/2 [==============================] - 1s 649ms/step - loss: 2.1482 - accuracy: 0.3400 - val_loss: 2.1939 - val_accuracy: 0.2511

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

# Evaluamos usando el test set
score = model1.evaluate(x_test, y_test, verbose=0)

print('Resultado en el test set:')
print('Test loss: {:0.4f}'.format(score[0]))
print('Test accuracy: {:0.2f}%'.format(score[1] * 100))

''' SALIDA CONSOLA
    [Nota: Saldrá un gráfico con 2 lineas donde 1 está más arriba que la otra.]
    Resultado en el test set:
    Test loss: 2.1939
    Test accuracy: 25.11%
    
'''

'''
    Como se puede ver en los resultados anteriores, parece que el modelo está haciendo overfitting: la varianza, diferencia 
    entre el error de entrenamiento y el de validación, es muy alta. Esto también se puede ver en el accuracy obtenido, 46% 
    para el conjunto de entrenamiento y 30% para la validación.
'''

#█▄▄ █░█ █▀ █▀▀ ▄▀█ █▀█   █▀ █▀█ █░░ █░█ █▀▀ █ █▀█ █▄░█   ▄▀█ █░░   █▀█ █▀█ █▀█ █▄▄ █░░ █▀▀ █▀▄▀█ ▄▀█
#█▄█ █▄█ ▄█ █▄▄ █▀█ █▀▄   ▄█ █▄█ █▄▄ █▄█ █▄▄ █ █▄█ █░▀█   █▀█ █▄▄   █▀▀ █▀▄ █▄█ █▄█ █▄▄ ██▄ █░▀░█ █▀█
#█▀▄ █▀▀ █░░   █▀█ █░█ █▀▀ █▀█ █▀▀ █ ▀█▀ ▀█▀ █ █▄░█ █▀▀
#█▄▀ ██▄ █▄▄   █▄█ ▀▄▀ ██▄ █▀▄ █▀░ █ ░█░ ░█░ █ █░▀█ █▄█


#█▀ █▀█ █░░ █░█ █▀▀ █ █▀█ █▄░█   ▄█ ▀   █▀▄▀█ █▀▀ ░░█ █▀█ █▀█ ▄▀█ █▀█   █▀▄▀█ █▀█ █▀▄ █▀▀ █░░ █▀█
#▄█ █▄█ █▄▄ █▄█ █▄▄ █ █▄█ █░▀█   ░█ ▄   █░▀░█ ██▄ █▄█ █▄█ █▀▄ █▀█ █▀▄   █░▀░█ █▄█ █▄▀ ██▄ █▄▄ █▄█
'''
Como el resultado obtenido con el modelo anterior no es muy bueno y además parece que está haciendo overfitting, vamos a crear 
otro modelo de red con más filtros por cada capa convolucional, y además le añadiremos un 20% de dropout.
'''

from tensorflow.keras.layers import Dropout

model2 = Sequential()

# Capa convolucional con 64 filtros de tamaño 3x3 seguida de un MaxPooling de 2x2
model2.add(Conv2D(64, (3, 3), activation='relu', input_shape=x_train.shape[1:]))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.2))

# Capa convolucional con 32 filtros de tamaño 3x3 seguida de un MaxPooling de 2x2
model2.add(Conv2D(32, (3, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.2))

# Capa Fully Connected
model2.add(Flatten())
model2.add(Dense(NUM_LABELS, activation='softmax'))

print(model2.summary())

# Compilamos y entrenamos
model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'] )
history = model2.fit(x_train, y_train, validation_data=(x_test, y_test),
                    batch_size=32, epochs=10, verbose=1)


plot_learning_curves(history)

# Evaluamos usando el test set
score = model2.evaluate(x_test, y_test, verbose=0)

print('Resultado en el test set:')
print('Test loss: {:0.4f}'.format(score[0]))
print('Test accuracy: {:0.2f}%'.format(score[1] * 100))

''' SALIDA CONSOLA
    Model: "sequential_8"
    _________________________________________________________________
    Layer (type)                Output Shape              Param #   
    =================================================================
    conv2d_13 (Conv2D)          (None, 26, 26, 64)        640       
                                                                    
    max_pooling2d_13 (MaxPoolin  (None, 13, 13, 64)       0         
    g2D)                                                            
                                                                    
    dropout_10 (Dropout)        (None, 13, 13, 64)        0         
                                                                    
    conv2d_14 (Conv2D)          (None, 11, 11, 32)        18464     
                                                                    
    max_pooling2d_14 (MaxPoolin  (None, 5, 5, 32)         0         
    g2D)                                                            
                                                                    
    dropout_11 (Dropout)        (None, 5, 5, 32)          0         
                                                                    
    flatten_8 (Flatten)         (None, 800)               0         
                                                                    
    dense_8 (Dense)             (None, 10)                8010      
                                                                    
    =================================================================
    Total params: 27,114
    Trainable params: 27,114
    Non-trainable params: 0
    _________________________________________________________________
    None
    Epoch 1/10
    2/2 [==============================] - 3s 2s/step - loss: 2.3070 - accuracy: 0.1000 - val_loss: 2.2902 - val_accuracy: 0.2156
    Epoch 2/10
    2/2 [==============================] - 1s 1s/step - loss: 2.2669 - accuracy: 0.2000 - val_loss: 2.2688 - val_accuracy: 0.2838
    Epoch 3/10
    2/2 [==============================] - 1s 1s/step - loss: 2.1958 - accuracy: 0.2200 - val_loss: 2.2499 - val_accuracy: 0.2881
    Epoch 4/10
    2/2 [==============================] - 1s 1s/step - loss: 2.1541 - accuracy: 0.2600 - val_loss: 2.2318 - val_accuracy: 0.2649
    Epoch 5/10
    2/2 [==============================] - 1s 1s/step - loss: 2.1318 - accuracy: 0.1800 - val_loss: 2.2122 - val_accuracy: 0.2602
    Epoch 6/10
    2/2 [==============================] - 1s 702ms/step - loss: 2.0895 - accuracy: 0.3600 - val_loss: 2.1870 - val_accuracy: 0.2663
    Epoch 7/10
    2/2 [==============================] - 1s 1s/step - loss: 2.0202 - accuracy: 0.3000 - val_loss: 2.1534 - val_accuracy: 0.2954
    Epoch 8/10
    2/2 [==============================] - 1s 691ms/step - loss: 1.9845 - accuracy: 0.3600 - val_loss: 2.1108 - val_accuracy: 0.3264
    Epoch 9/10
    2/2 [==============================] - 1s 1s/step - loss: 1.8758 - accuracy: 0.4000 - val_loss: 2.0584 - val_accuracy: 0.3594
    Epoch 10/10
    2/2 [==============================] - 1s 704ms/step - loss: 1.8106 - accuracy: 0.4600 - val_loss: 1.9968 - val_accuracy: 0.3880
    
    
    Resultado en el test set:
    Test loss: 1.9968
    Test accuracy: 38.80%
'''


'''
Con esta nueva topología de red hemos conseguido mejorar el resultado, además la varianza obtenida es mucho más baja (ya no está haciendo 
overfitting). Sin embargo, el error de entrenamiento (el bias) sigue siendo bastante alto.

Para solucionar esto podríamos aplicar aumentado de datos (este ejercicio se deja como opcional) o añadir más datos al conjunto de 
entrenamiento. Como inicialmente habíamos limitado los datos vamos a usar la segunda estrategia.
'''

#█▀ █▀█ █░░ █░█ █▀▀ █ █▀█ █▄░█   ▀█ ▀   █▀▄▀█ ▄▀█ █▀   █▀▄ ▄▀█ ▀█▀ █▀█ █▀
#▄█ █▄█ █▄▄ █▄█ █▄▄ █ █▄█ █░▀█   █▄ ▄   █░▀░█ █▀█ ▄█   █▄▀ █▀█ ░█░ █▄█ ▄█

'''
En este paso vamos a restaurar todas las imágenes de entrenamiento que nos habíamos guardado al principio en las variables `x_train_backup` 
y `y_train_backup`, y volveremos a lanzar el entrenamiento para el segundo modelo de red que hemos creado.
'''

# Restauramos todas las imágenes de entrenamiento
x_train = x_train_backup
y_train = y_train_backup

print('Datos para entrenamiento:')
print(' - x_train: {}'.format( x_train.shape ))
print(' - y_train: {}'.format( y_train.shape ))


# Iniciamos el entrenamiento
history = model2.fit(x_train, y_train, validation_data=(x_test, y_test),
                    batch_size=32, epochs=10, verbose=1)

plot_learning_curves(history)


# Evaluamos usando el test set
score = model2.evaluate(x_test, y_test, verbose=0)

print('Resultado en el test set:')
print('Test loss: {:0.4f}'.format(score[0]))
print('Test accuracy: {:0.2f}%'.format(score[1] * 100))

'''  SALIDA CONSOLA

    Datos para entrenamiento:
    - x_train: (60000, 28, 28, 1)
    - y_train: (60000, 10)
    Epoch 1/10
    1875/1875 [==============================] - 9s 5ms/step - loss: 0.5249 - accuracy: 0.8099 - val_loss: 0.3936 - val_accuracy: 0.8615
    Epoch 2/10
    1875/1875 [==============================] - 9s 5ms/step - loss: 0.3774 - accuracy: 0.8643 - val_loss: 0.3531 - val_accuracy: 0.8735
    Epoch 3/10
    1875/1875 [==============================] - 10s 5ms/step - loss: 0.3401 - accuracy: 0.8759 - val_loss: 0.3367 - val_accuracy: 0.8789
    Epoch 4/10
    1875/1875 [==============================] - 10s 5ms/step - loss: 0.3162 - accuracy: 0.8867 - val_loss: 0.3280 - val_accuracy: 0.8806
    Epoch 5/10
    1875/1875 [==============================] - 8s 4ms/step - loss: 0.2989 - accuracy: 0.8913 - val_loss: 0.2979 - val_accuracy: 0.8917
    Epoch 6/10
    1875/1875 [==============================] - 10s 5ms/step - loss: 0.2866 - accuracy: 0.8944 - val_loss: 0.3092 - val_accuracy: 0.8895
    Epoch 7/10
    1875/1875 [==============================] - 10s 5ms/step - loss: 0.2798 - accuracy: 0.8984 - val_loss: 0.2805 - val_accuracy: 0.8990
    Epoch 8/10
    1875/1875 [==============================] - 10s 5ms/step - loss: 0.2705 - accuracy: 0.9004 - val_loss: 0.2789 - val_accuracy: 0.8950
    Epoch 9/10
    1875/1875 [==============================] - 9s 5ms/step - loss: 0.2657 - accuracy: 0.9026 - val_loss: 0.2961 - val_accuracy: 0.8917
    Epoch 10/10
    1875/1875 [==============================] - 9s 5ms/step - loss: 0.2610 - accuracy: 0.9038 - val_loss: 0.2742 - val_accuracy: 0.8994

    Resultado en el test set:
    Test loss: 0.2742
    Test accuracy: 89.94%
'''

'''
Como se puede ver, al entrenar con muchos más datos el resultado obtenido ha mejorado hasta alcanzar el 90% de acierto.
'''
