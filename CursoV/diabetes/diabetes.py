import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential, layers
import matplotlib.pyplot as plt

#█▀█ ▄▀█ █▀ █▀█   ▄█ ▀   █▀█ █▄▄ ▀█▀ █▀▀ █▄░█ █▀▀ █▀█   █▄█   ▀█▀ █▀█ ▄▀█ █▄░█ █▀ █▀▀ █▀█ █▀█ █▀▄▀█ ▄▀█ █▀█
#█▀▀ █▀█ ▄█ █▄█   ░█ ▄   █▄█ █▄█ ░█░ ██▄ █░▀█ ██▄ █▀▄   ░█░   ░█░ █▀▄ █▀█ █░▀█ ▄█ █▀░ █▄█ █▀▄ █░▀░█ █▀█ █▀▄

#█▀▄ ▄▀█ ▀█▀ █▀█ █▀
#█▄▀ █▀█ ░█░ █▄█ ▄█
# Clasificación binaria con el fichero 'diabete_01.csv'
data = pd.read_csv('diabetes_01.csv')

''' Los datos en diabetes_01.csv son:
preg,plas,pres,skin,insu,mass,pedi,age,class
6, 148 , 72, 35 ,0 ,33.6 ,0.627 ,50, 1
1, 85 , 66, 29, 0, 26.6, 0.351, 31, 0
8, 183, 64, 0, 0, 23.3, 0.672, 32, 1
etc
NOTA: Class indica si esos datos indican diabetes(1) o no(0)
'''

pairs = {'int64':'int32', 'float64':'float32'}
for i in data.columns:
  data[i]= data[i].astype(pairs.get(str(data[i].dtype), data[i].dtype))

# Selección de atributos y variable objetivo
X = data.iloc[:,:-1] #Escojo todas las filas y todas las columnas menos la última columna 
y = data.iloc[:,-1] #Escojo todas las filas y solo la última columna(class)
'''
[x,y] ->[filas,columnas] | [:,:] -> todas las filas y todas las columnas
: -> todas , :-1 -> todas menos una , -1 -> solo la última
'''

# Dividir los datos en entrenamiento y test
#El modelo se entrenará con x_train/y_train y una vez finalizado el entrenamiento
#se usará x_test/y_test que es un conjunto de datos diferente al del entrenamiento
#para evaluar cuánto de fiable es el modelo en sus predicciones con datos nuevos.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=123)


#-------------------------------------------------------------------------------
print(f' - Cantidad de filas de datos: {X_train.shape}')
print(f' - Cantidad de etiquetas: {y_train.shape}')

''' SALIDA CONSOLA
 - Cantidad de filas de datos: (691, 8)
 - Cantidad de etiquetas: (691,)
'''
#-------------------------------------------------------------------------------
#NOTA: En x_train está el dato en forma de excel con los valores
#      que tiene cada columna. 
#      En y_train está 0 ó 1 dependiendo de si los datos que hay 
#      en x_train son indicativos de que tiene diabetes.
#La idea es que si le pasamos a un modelo entrenado con estos datos
#otros datos distintos sepa decir si tiene (1) o no (0) diabetes.
print(f' - X_train[0]: {X_train.iloc[0]}')
print(f' - y_train[0]: {y_train.iloc[0]}')

''' SALIDA CONSOLA
 - X_train[0]: preg      6.000  
plas    125.000
pres     78.000
skin     31.000
insu      0.000
mass     27.600
pedi      0.565
age      49.000
Name: 701, dtype: float64

 - y_train[0]: 1
'''
#---------------------------------------------------------------------------------

'''
Cuando hay un capa que en la siguiente linea tiene un droput significa que el x porciento
(0.2 -> 20%, 0.3 ->30% , etc) de neuronas (elegidas al azar) serán descartadas durante el 
entrenamiento para que el modelo mejore la generalización (es decir, que no se acostumbre 
a predecir con las  mismas neuronas). Esto en teoría tiene que mejorar el rendimiento del 
modelo.
'''
#█▀█ ▄▀█ █▀ █▀█   ▀█ ▀   █▀▄ █▀▀ █▀▀ █ █▄░█ █ █▀█   ▄▀█ █▀█ █▀█ █░█ █ ▀█▀ █▀▀ █▀▀ ▀█▀ █░█ █▀█ ▄▀█
#█▀▀ █▀█ ▄█ █▄█   █▄ ▄   █▄▀ ██▄ █▀░ █ █░▀█ █ █▀▄   █▀█ █▀▄ ▀▀█ █▄█ █ ░█░ ██▄ █▄▄ ░█░ █▄█ █▀▄ █▀█
model = Sequential([
    layers.Dense(32, input_dim=X_train.shape[1], activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

print(model.summary())

''' SALIDA CONSOLA
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 dense (Dense)               (None, 32)                288

 dropout (Dropout)           (None, 32)                0

 dense_1 (Dense)             (None, 16)                528

 dropout_1 (Dropout)         (None, 16)                0

 dense_2 (Dense)             (None, 1)                 17

=================================================================
Total params: 833 (3.25 KB)
Trainable params: 833 (3.25 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
None
'''

#█▀█ ▄▀█ █▀ █▀█  ▀█ ▀   █▀▀ █▄░█ ▀█▀ █▀█ █▀▀ █▄░█ ▄▀█ █▀█   █▀▄▀█ █▀█ █▀▄ █▀▀ █░░ █▀█
#█▀▀ █▀█ ▄█ █▄█  ▄█ ▄   ██▄ █░▀█ ░█░ █▀▄ ██▄ █░▀█ █▀█ █▀▄   █░▀░█ █▄█ █▄▀ ██▄ █▄▄ █▄█
model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=70, batch_size=8)

'''
87/87 [==============================] - 1s 849us/step - loss: 2.3036 - accuracy: 0.6151
Epoch 2/70
87/87 [==============================] - 0s 767us/step - loss: 0.7028 - accuracy: 0.6397
Epoch 3/70
87/87 [==============================] - 0s 802us/step - loss: 0.6735 - accuracy: 0.6324
Epoch 4/70
87/87 [==============================] - 0s 774us/step - loss: 0.6692 - accuracy: 0.6541
Epoch 5/70
87/87 [==============================] - 0s 744us/step - loss: 0.6548 - accuracy: 0.6512
Epoch 6/70
...
87/87 [==============================] - 0s 733us/step - loss: 0.6417 - accuracy: 0.6556
Epoch 46/70
87/87 [==============================] - 0s 744us/step - loss: 0.6347 - accuracy: 0.6570
...
87/87 [==============================] - 0s 756us/step - loss: 0.6389 - accuracy: 0.6527
Epoch 66/70
87/87 [==============================] - 0s 791us/step - loss: 0.6242 - accuracy: 0.6541
Epoch 67/70
87/87 [==============================] - 0s 744us/step - loss: 0.6271 - accuracy: 0.6541
Epoch 68/70
87/87 [==============================] - 0s 756us/step - loss: 0.6344 - accuracy: 0.6556
Epoch 69/70
87/87 [==============================] - 0s 744us/step - loss: 0.6244 - accuracy: 0.6556
Epoch 70/70
87/87 [==============================] - 0s 733us/step - loss: 0.6313 - accuracy: 0.6512
3/3 [==============================] - 0s 2ms/step - loss: 0.6782 - accuracy: 0.6104
Test accuracy 0.61
'''

#█▀█ ▄▀█ █▀ █▀█   █░█ ▀   █▀█ █▀▀ █▀ █░█ █░░ ▀█▀ ▄▀█ █▀▄ █▀█ █▀   █▄█   █▀█ █▀█ █▀▀ █▀▄ █ █▀▀ █▀▀ █ █▀█ █▄░█
#█▀▀ █▀█ ▄█ █▄█   ▀▀█ ▄   █▀▄ ██▄ ▄█ █▄█ █▄▄ ░█░ █▀█ █▄▀ █▄█ ▄█   ░█░   █▀▀ █▀▄ ██▄ █▄▀ █ █▄▄ █▄▄ █ █▄█ █░▀█

loss, accuracy = model.evaluate(X_test, y_test, batch_size=32)
print(f'Test accuracy {accuracy:.2f}')