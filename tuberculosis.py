import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report

import os
import numpy as np
import cv2
import tensorflow as tf

#█▀█ ▄▀█ █▀ █▀█   ▄█ ▀   █▀█ █▄▄ ▀█▀ █▀▀ █▄░█ █▀▀ █▀█   █▄█   ▀█▀ █▀█ ▄▀█ █▄░█ █▀ █▀▀ █▀█ █▀█ █▀▄▀█ ▄▀█ █▀█
#█▀▀ █▀█ ▄█ █▄█   ░█ ▄   █▄█ █▄█ ░█░ ██▄ █░▀█ ██▄ █▀▄   ░█░   ░█░ █▀▄ █▀█ █░▀█ ▄█ █▀░ █▄█ █▀▄ █░▀░█ █▀█ █▀▄

#█▀▄ ▄▀█ ▀█▀ █▀█ █▀
#█▄▀ █▀█ ░█░ █▄█ ▄█

carpeta_principal = "dataset_tuberculosis" # Carpeta contenedora  de los datasets

def procesar_imagenes(subcarpeta):
    imagenes = [] 
    etiquetas = []
    subcarpeta_dir = os.path.join(carpeta_principal, subcarpeta) #'dataset_tuberculosis/Normal' y 'dataset_tuberculosis/Tuberculosis'
    i=0 if subcarpeta == "Normal" else 1
    
    for filename in os.listdir(subcarpeta_dir): #recorro todas las fotos
        if filename.endswith(".png"):
            img_path = os.path.join(subcarpeta_dir, filename) #'dataset_tuberculosis/Normal/Normal-1.png'
            img_grises = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)# Leer la imagen en escala de grises
            #img_grises = cv2.resize(img_grises, (width, height)) # Redimensionar la imagen si es necesario    
            imagenes.append(img_grises)  # Almacenar la imagen
            etiquetas.append(i)  #Añado etiqueta

    return imagenes, etiquetas

def normalizar_imagenes(imagenes,etiquetas):
    
    # Convertir las listas a arrays numpy
    x_train = np.array(imagenes)
    y_train = np.array(etiquetas)
    
    x_train = x_train.astype('float32') / 255.0 #Normalizar los valores de píxeles de las imágenes
    
    # Reajustar las dimensiones del array de imágenes para que coincida con el formato esperado por la CNN
    x_train = np.expand_dims(x_train, axis=-1)
    
    return x_train,y_train

imagenes_normal , etiquetas_normal = procesar_imagenes("Normal")
imagenes_tuberculosis , etiquetas_tuberculosis = procesar_imagenes("Tuberculosis")

x_train_normal , y_train_normal = normalizar_imagenes(imagenes_normal, etiquetas_normal)
x_train_tuberculosis , y_train_tuberculosis = normalizar_imagenes(imagenes_tuberculosis, etiquetas_tuberculosis)

# Dividir datos de la clase Normal en entrenamiento y evaluación
x_train_normal, x_test_normal, y_train_normal, y_test_normal = train_test_split(x_train_normal, y_train_normal, test_size=0.2, random_state=42, stratify=y_train_normal)

# Dividir datos de la clase Tuberculosis en entrenamiento y evaluación
x_train_tuberculosis, x_test_tuberculosis, y_train_tuberculosis, y_test_tuberculosis = train_test_split(x_train_tuberculosis, y_train_tuberculosis, test_size=0.2, random_state=42, stratify=y_train_tuberculosis)

# Concatenar los conjuntos de evaluación de ambas clases
x_test = np.concatenate((x_test_normal, x_test_tuberculosis), axis=0)
y_test = np.concatenate((y_test_normal, y_test_tuberculosis), axis=0)

# Concatenar los conjuntos de entrenamiento de ambas clases
x_train = np.concatenate((x_train_normal, x_train_tuberculosis), axis=0)
y_train = np.concatenate((y_train_normal, y_train_tuberculosis), axis=0)

# Visualizamos una imagen con su etiqueta
    #plt.imshow(np.squeeze(x_train[0]), cmap='gray')
    #plt.title(f'Etiqueta: {etiquetas[0]}')
    #plt.axis('off')
    #plt.show()
    
# Visualizar cantidad imágenes de cada clase
#plt.figure(figsize=(8, 6))
#plt.bar(["Normal", "Tuberculosis"], [len(x_train_normal), len(x_train_tuberculosis)], color=['Green', 'red'])
#plt.title("Distribución de clase")
#plt.xlabel("Clase")
#plt.ylabel("Cantidad de imágenes")
#plt.show()

# Dimensiones de los conjuntos de datos
# Imprimir la forma de los arrays
print("Forma de x_train_normal:", x_train_normal.shape)
print("Forma de y_train_normal:", y_train_normal.shape)
print("Forma de x_train_tuberculosis:", x_train_tuberculosis.shape)
print("Forma de y_train_tuberculosis:", y_train_tuberculosis.shape)

print(f'Tamaño del conjunto de entrenamiento: {len(x_train)}')
print(f'Tamaño del conjunto de test: {len(x_test)}')

#█▀█ ▄▀█ █▀ █▀█   ▀█ ▀   █▀▄ █▀▀ █▀▀ █ █▄░█ █ █▀█   ▄▀█ █▀█ █▀█ █░█ █ ▀█▀ █▀▀ █▀▀ ▀█▀ █░█ █▀█ ▄▀█
#█▀▀ █▀█ ▄█ █▄█   █▄ ▄   █▄▀ ██▄ █▀░ █ █░▀█ █ █▀▄   █▀█ █▀▄ ▀▀█ █▄█ █ ░█░ ██▄ █▄▄ ░█░ █▄█ █▀▄ █▀█


'''
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(512, 512, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))  # Para evitar/reducir overfitting
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid')) 
'''
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(512, 512, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(1, activation='sigmoid')
])

#█▀█ ▄▀█ █▀ █▀█  ▀█ ▀   █▀▀ █▄░█ ▀█▀ █▀█ █▀▀ █▄░█ ▄▀█ █▀█   █▀▄▀█ █▀█ █▀▄ █▀▀ █░░ █▀█
#█▀▀ █▀█ ▄█ █▄█  ▄█ ▄   ██▄ █░▀█ ░█░ █▀▄ ██▄ █░▀█ █▀█ █▀▄   █░▀░█ █▄█ █▄▀ ██▄ █▄▄ █▄█

# Compilar el modelo
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Resumen del modelo
model.summary()

history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

#█▀█ ▄▀█ █▀ █▀█   █░█ ▀   █▀█ █▀▀ █▀ █░█ █░░ ▀█▀ ▄▀█ █▀▄ █▀█ █▀   █▄█   █▀█ █▀█ █▀▀ █▀▄ █ █▀▀ █▀▀ █ █▀█ █▄░█
#█▀▀ █▀█ ▄█ █▄█   ▀▀█ ▄   █▀▄ ██▄ ▄█ █▄█ █▄▄ ░█░ █▀█ █▄▀ █▄█ ▄█   ░█░   █▀▀ █▀▄ ██▄ █▄▀ █ █▄▄ █▄▄ █ █▄█ █░▀█

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, 1])
plt.legend(loc='upper right')
plt.show()

# Model prediction
y_pred_probs = model.predict(x_test) 
y_pred_labels = (y_pred_probs > 0.5).astype(int) 
print(classification_report(y_test, y_pred_labels, digits=4, zero_division=0))


#█▀▀ █░█ ▄▀█ █▀█ █▀▄ ▄▀█ █▀█   █▀▄▀█ █▀█ █▀▄ █▀▀ █░░ █▀█
#█▄█ █▄█ █▀█ █▀▄ █▄▀ █▀█ █▀▄   █░▀░█ █▄█ █▄▀ ██▄ █▄▄ █▄█

model.save('tb_modelo_2')


#█▀▀ ▄▀█ █▀█ █▀▀ ▄▀█ █▀█   █▀▄▀█ █▀█ █▀▄ █▀▀ █░░ █▀█
#█▄▄ █▀█ █▀▄ █▄█ █▀█ █▀▄   █░▀░█ █▄█ █▄▀ ██▄ █▄▄ █▄█


