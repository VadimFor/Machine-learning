import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential, layers, optimizers, utils

#█▀█ ▄▀█ █▀ █▀█   ▄█ ▀   █▀█ █▄▄ ▀█▀ █▀▀ █▄░█ █▀▀ █▀█   █▄█   ▀█▀ █▀█ ▄▀█ █▄░█ █▀ █▀▀ █▀█ █▀█ █▀▄▀█ ▄▀█ █▀█
#█▀▀ █▀█ ▄█ █▄█   ░█ ▄   █▄█ █▄█ ░█░ ██▄ █░▀█ ██▄ █▀▄   ░█░   ░█░ █▀▄ █▀█ █░▀█ ▄█ █▀░ █▄█ █▀▄ █░▀░█ █▀█ █▀▄

#█▀▄ ▄▀█ ▀█▀ █▀█ █▀
#█▄▀ █▀█ ░█░ █▄█ ▄█

data = pd.read_csv('iris.csv')

''' Los datos en diabetes_01.csv son:
sepallength,sepalwidth,petallength,petalwidth,class
5.1,3.5,1.4,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
7.0,3.2,4.7,1.4,Iris-versicolor
6.4,3.2,4.5,1.5,Iris-versicolor
6.3,3.3,6.0,2.5,Iris-virginica
5.8,2.7,5.1,1.9,Iris-virginica
etc
NOTA: Class indica el tipo que representan la combinación de columnas anteriores.
'''

# Selección de atributos y variable objetivo -------------------------
X = data.iloc[:,:-1] #Escojo todas las filas y todas las columnas menos la última columna 
#------------------------------------------------------------------------
'''
Escojo todas las filas y solo la última columna(class) y lo transformo a one-hot.
La idea de hacer esto es que keras tiene que recibir datos en forma de bits 1 ó 0.
En el ejercicio de diabetes solo hay 1 ó 0 dependiendo de si tiene diabetes o no, pero
en este caso como hay 3 respuestas posibles: 
    -0 para iris-setosa, 
    -1 para iris.verginica
    -2 para iris-versicolor
entonces la única forma de pasarlo a modo one-hot sabiendo que pueden haber 3 respuestas
posibles es utilizando 3 bits:
    -0 -> [1,0,0]
    -1 -> [0,1,0]
    -2 -> [0,0,1]
Antes de nada lo primero que hay que hacer es transformar los nombres de la clases a
0 o 1 o 2 mediante labelEncoder. Y más adelante a one hot (0 0 0) con to_categorical
'''
label_enc = LabelEncoder()

'''
[x,y] ->[filas,columnas] | [:,:] -> todas las filas y todas las columnas
: -> todas , :-1 -> todas menos una , -1 -> solo la última
'''
y = label_enc.fit_transform(data.iloc[:,-1]) #Transforma a 0,1,2,etc (dependiende de cuantas clases hay)
cantidad_clases = len(np.unique(y)) #Cuantas clases diferentes hay (aqúi es 3)
#------------------------------------------------------------------------

# Dividir los datos en entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=123) #test_size = .1 significa 10% (.2 = 20% , .3 =30% , etc)
y_train = utils.to_categorical(y_train) #paso las etiquetas a ONE-HOT [1 0 0] [0 1 0] [0 0 1]
y_test = utils.to_categorical(y_test)   #paso las etiquetas a ONE-HOT [1 0 0] [0 1 0] [0 0 1]

#-------------------------------------------------------------------------------
print(f' - Cantidad de filas de datos: {X_train.shape}')
print(f' - Cantidad de etiquetas: {y_train.shape}')

''' SALIDA CONSOLA
 - Cantidad de filas de datos: (135, 4)
 - Cantidad de etiquetas: (135, 3)
'''
#-------------------------------------------------------------------------------
print(f' - X_train[0]: {X_train.iloc[0]}')
print(f' - y_train[0]: {y_train[0]}')

''' SALIDA CONSOLA
 - X_train[0]: sepallength    6.5
sepalwidth     3.0
petallength    5.8
petalwidth     2.2
Name: 104, dtype: float64

 - y_train[0]: [0. 0. 1.]
'''
#---------------------------------------------------------------------------------

#█▀█ ▄▀█ █▀ █▀█   ▀█ ▀   █▀▄ █▀▀ █▀▀ █ █▄░█ █ █▀█   ▄▀█ █▀█ █▀█ █░█ █ ▀█▀ █▀▀ █▀▀ ▀█▀ █░█ █▀█ ▄▀█
#█▀▀ █▀█ ▄█ █▄█   █▄ ▄   █▄▀ ██▄ █▀░ █ █░▀█ █ █▀▄   █▀█ █▀▄ ▀▀█ █▄█ █ ░█░ ██▄ █▄▄ ░█░ █▄█ █▀▄ █▀█
'''
Cuando hay un capa que en la siguiente linea tiene un droput significa que el x porciento
(0.2 -> 20%, 0.3 ->30% , etc) de neuronas (elegidas al azar) serán descartadas durante el 
entrenamiento para que el modelo mejore la generalización (es decir, que no se acostumbre 
a predecir con las  mismas neuronas). Esto en teoría tiene que mejorar el rendimiento del 
modelo.
'''
model = Sequential([
  # Dense(64) es una capa de conexión completa con 64 neuronas ocultas.
  # en la primera capa se tiene que especificar la dimensión de la entrada de datos.
  layers.Dense(64, activation='relu', input_dim=X_train.shape[1]),
  layers.Dropout(0.2),
  layers.Dense(64, activation='relu'),
  layers.Dropout(0.3),
  layers.Dense(cantidad_clases, activation='softmax') # Cantidad de clases finales
])

print(model.summary())

''' SALIDA CONSOLA
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 dense (Dense)               (None, 64)                320

 dropout (Dropout)           (None, 64)                0

 dense_1 (Dense)             (None, 64)                4160

 dropout_1 (Dropout)         (None, 64)                0

 dense_2 (Dense)             (None, 3)                 195

=================================================================
Total params: 4675 (18.26 KB)
Trainable params: 4675 (18.26 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
None
'''

#█▀█ ▄▀█ █▀ █▀█  ▀█ ▀   █▀▀ █▄░█ ▀█▀ █▀█ █▀▀ █▄░█ ▄▀█ █▀█   █▀▄▀█ █▀█ █▀▄ █▀▀ █░░ █▀█
#█▀▀ █▀█ ▄█ █▄█  ▄█ ▄   ██▄ █░▀█ ░█░ █▀▄ ██▄ █░▀█ █▀█ █▀▄   █░▀░█ █▄█ █▄▀ ██▄ █▄▄ █▄█

''' [SGD = Stochastic Gradient Descent]
En el contexto del entrenamiento de modelos de aprendizaje profundo, como las redes 
neuronales, el uso de un optimizador es fundamental pero no necesariamente obligatorio. 
Sin embargo, es altamente recomendado, ya que sin un optimizador, el modelo no tendría 
una forma de ajustar sus parámetros durante el entrenamiento para minimizar la función 
de pérdida y mejorar su rendimiento.

El momentum y Nesterov son técnicas diseñadas para ayudar al optimizador SGD a converger más rápido y a 
evitar ciertos problemas durante el entrenamiento, como oscilaciones alrededor de mínimos locales. Sin 
embargo, en algunos casos, pueden no ser necesarios o incluso contraproducentes.
Si bien el uso de momentum y Nesterov puede ser beneficioso en muchos casos, no son requisitos obligatorios 
para utilizar el optimizador SGD. La necesidad de utilizar momentum y Nesterov depende del problema específico 
que estés abordando y de la estructura de tus datos.
Cuándo es bueno utilizar momentum y Nesterov:
  -Cuando se está experimentando overfitting debido a pocos datos.
  -Conjuntos de daots muy grandes y complejos con pérdida irregular.
  
Aunque lo mejor es siempre ir experimentando 'con' y 'sin' momento/nestervo y ver cual dá mejor resultado.

IMPORTANTE
---------------------------------------------------------------------------------------------------------
Aquí se puede elegir si utilizar el SGD por defecto (optimizer='sgd') o utilizarlo como se está utilizando
aquí, pero si se pone por defecto el accurracy baja de 0.88 (con momentum/nesterov) a 0.66(por defecto)

'''
sgd = optimizers.SGD( momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20, batch_size=32)


#█▀█ ▄▀█ █▀ █▀█   █░█ ▀   █▀█ █▀▀ █▀ █░█ █░░ ▀█▀ ▄▀█ █▀▄ █▀█ █▀   █▄█   █▀█ █▀█ █▀▀ █▀▄ █ █▀▀ █▀▀ █ █▀█ █▄░█
#█▀▀ █▀█ ▄█ █▄█   ▀▀█ ▄   █▀▄ ██▄ ▄█ █▄█ █▄▄ ░█░ █▀█ █▄▀ █▄█ ▄█   ░█░   █▀▀ █▀▄ ██▄ █▄▀ █ █▄▄ █▄▄ █ █▄█ █░▀█

loss, accuracy = model.evaluate(X_test, y_test, batch_size=32)
print(f'Test accuracy {accuracy:.2f}')