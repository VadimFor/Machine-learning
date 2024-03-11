import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

tf.random.set_seed(1)  # Fijamos la semilla de TF
np.random.seed(1)  # Fijamos la semilla

# Descargamos la base de datos
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Mostramos algunas imágenes
n = 15
index = np.random.randint(len(x_train), size=n)
plt.figure(figsize=(n*1.5, 1.5))
for i in np.arange(n):
    ax = plt.subplot(1,n,i+1)
    ax.set_title('{} ({})'.format(y_train[index[i]],index[i]))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.imshow(x_train[index[i]], cmap='gray')
plt.show()

# Mostramos las dimensiones de los datos
print('Datos para entrenamiento:')
print(' - x_train: {}'.format(str(x_train.shape)))
print(' - y_train: {}'.format(str(y_train.shape)))
print('Datos para evaluación:')
print(' - x_test: {}'.format(str(x_test.shape)))
print(' - y_test: {}'.format(str(y_test.shape)))


"""
Preparamos los datos para la red
"""

# Redimensionamos para añadir el canal
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

# Transformamos a decimal
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

# Normalizamos entre 0 y 1
x_train /= 255.
x_test /= 255.

# Transformamos las etiquetas a categórico (one-hot)
NUM_LABELS = 10
y_train  = tf.keras.utils.to_categorical(y_train, NUM_LABELS)
y_test = tf.keras.utils.to_categorical(y_test, NUM_LABELS)


# Mostramos (de nuevo) las dimensiones de los datos
print('Datos para entrenamiento:')
print(' - x_train: {}'.format(str(x_train.shape)))
print(' - y_train: {}'.format(str(y_train.shape)))
print('Datos para evaluación:')
print(' - x_test: {}'.format(str(x_test.shape)))
print(' - y_test: {}'.format(str(y_test.shape)))


"""
Definimos la CNN a utilizar y la entrenamos
"""

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

model = Sequential()

# Capa convolucional con 8 filtros de tamaño 3x3 seguida de un MaxPooling de 2x2
model.add(Conv2D(8, (3, 3), activation='relu', name='conv1', input_shape=x_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Capa convolucional con 4 filtros de tamaño 3x3 seguida de un MaxPooling de 2x2
model.add(Conv2D(4, (3, 3), activation='relu', name='conv2'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Capa Fully Connected de salida con función de activación Softmax
model.add(Flatten())
model.add(Dense(NUM_LABELS, activation='softmax'))

# Mostramos el resumen de la red
print(model.summary())

# La compilamos usando "categorical crossentropy" como función de pérdida,
# Adam como optimizador y añadimos la métrica accuracy
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'] )

# Iniciamos el entrenamiento durante 5 épocas con un tamaño de batch de 32
history = model.fit(x_train, y_train, validation_split=0.33,
                    batch_size=32, epochs=5, verbose=1)



"""
Mostramos las curvas de aprendizaje y evaluamos usando el test set
"""

# -----------------------------
def plot_learning_curves(hist):
  plt.plot(hist.history['loss'])
  plt.plot(hist.history['val_loss'])
  plt.title('Curvas de aprendizaje')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Conjunto de entrenamiento', 'Conjunto de validación'], loc='upper right')
  plt.show()

print('Mostramos las curvas de aprendizaje')
plot_learning_curves(history)


# Evaluamos usando el test set
score = model.evaluate(x_test, y_test, verbose=0)

print('Resultado en el test set:')
print('Test loss: {:0.4f}'.format(score[0]))
print('Test accuracy: {:0.2f}%'.format(score[1] * 100))


"""
Mostramos los filtros aprendidos por la red
"""

# --------------------------------
def plot_figures(images):
  width = images.shape[0]
  n_filters = images.shape[2]
  plt.figure(figsize=(1.5 * n_filters, 1.5))
  for i in range(n_filters):
    ax = plt.subplot(1,n_filters,i+1)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.imshow(np.array(images[:,:,i] * 255., dtype=np.uint8), cmap='gray')
  plt.show()


print('Filtros aprendidos por la primera capa:')
modelConv = Model(inputs=model.input, outputs=model.get_layer("conv1").output)
predictions = modelConv.predict(x_train)
print(predictions.shape)
plot_figures(predictions[0])
plot_figures(predictions[1])
plot_figures(predictions[2])

print('Filtros aprendidos por la segunda capa:')
modelConv = Model(inputs=model.input, outputs=model.get_layer("conv2").output)
predictions = modelConv.predict(x_train)
print(predictions.shape)
plot_figures(predictions[0])
plot_figures(predictions[1])
plot_figures(predictions[2])


print('Valores del primer filtro aprendido para la segunda capa:')
print( model.get_layer("conv2").get_weights()[0][:,:,0,0] )
