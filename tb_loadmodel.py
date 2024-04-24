import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from PIL import Image
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

# Imprimir la forma de los arrays
print("Forma de x_train_normal:", x_train_normal.shape)
print("Forma de y_train_normal:", y_train_normal.shape)
print("Forma de x_train_tuberculosis:", x_train_tuberculosis.shape)
print("Forma de y_train_tuberculosis:", y_train_tuberculosis.shape)

# Dividir datos de la clase Normal en entrenamiento y evaluación
x_train_normal, x_eval_normal, y_train_normal, y_eval_normal = train_test_split(x_train_normal, y_train_normal, test_size=0.2, random_state=42, stratify=y_train_normal)

# Dividir datos de la clase Tuberculosis en entrenamiento y evaluación
x_train_tuberculosis, x_eval_tuberculosis, y_train_tuberculosis, y_eval_tuberculosis = train_test_split(x_train_tuberculosis, y_train_tuberculosis, test_size=0.2, random_state=42, stratify=y_train_tuberculosis)

# Concatenar los conjuntos de evaluación de ambas clases
x_test = np.concatenate((x_eval_normal, x_eval_tuberculosis), axis=0)
y_test = np.concatenate((y_eval_normal, y_eval_tuberculosis), axis=0)

# Concatenar los conjuntos de entrenamiento de ambas clases
x_train = np.concatenate((x_train_normal, x_train_tuberculosis), axis=0)
y_train = np.concatenate((y_train_normal, y_train_tuberculosis), axis=0)


model = load_model('tb_modelo_2')
metricas = model.evaluate(x_test, y_test, verbose=0, return_dict=True)
for k, v in metricas.items():
    print(f'{k}: {v:.4f}')
    
model.summary()


# Model prediction

y_pred_probs = model.predict(x_test) 
y_pred_labels = (y_pred_probs > 0.5).astype(int)

print(classification_report(y_test, y_pred_labels, digits=4, zero_division=0))

