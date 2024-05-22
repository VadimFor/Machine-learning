from sklearn.model_selection import train_test_split
import cv2
import os
import numpy as np

import matplotlib.pyplot as plt
import visualizacion

from sklearn.utils import class_weight

import random

######################################################################################################################################################
#█▀█ █▄▄ ▀█▀ █▀▀ █▄░█ █▀▀ █▀█   █▄█   ▀█▀ █▀█ ▄▀█ █▄░█ █▀ █▀▀ █▀█ █▀█ █▀▄▀█ ▄▀█ █▀█   █▀▄ ▄▀█ ▀█▀ █▀█ █▀
#█▄█ █▄█ ░█░ ██▄ █░▀█ ██▄ █▀▄   ░█░   ░█░ █▀▄ █▀█ █░▀█ ▄█ █▀░ █▄█ █▀▄ █░▀░█ █▀█ █▀▄   █▄▀ █▀█ ░█░ █▄█ ▄█
carpeta_principal = "C:/Users/vadim/Desktop/TFG/dataset_Tuberculosis" # Carpeta contenedora  de los datasets

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



def obtener_train_test(dimensiones = 0):
    
    imagenes_normal , etiquetas_normal = procesar_imagenes("Normal")
    imagenes_tuberculosis , etiquetas_tuberculosis = procesar_imagenes("Tuberculosis")

    x_train_normal , y_train_normal = normalizar_imagenes(imagenes_normal, etiquetas_normal)
    x_train_tuberculosis , y_train_tuberculosis = normalizar_imagenes(imagenes_tuberculosis, etiquetas_tuberculosis)

    # Divido 80 (train) - 20  para el conjunto de fotos Normal
    x_train_normal, x_test_normal, y_train_normal, y_test_normal = train_test_split(x_train_normal, y_train_normal, test_size=0.2, random_state=42, stratify=y_train_normal)

    # Divido 80 (train) - 20  para el conjunto de fotos Tuberculosis
    x_train_tuberculosis, x_test_tuberculosis, y_train_tuberculosis, y_test_tuberculosis = train_test_split(x_train_tuberculosis, y_train_tuberculosis, test_size=0.2, random_state=42, stratify=y_train_tuberculosis)

    #▀█▀ ▄▀█ █▀█ █▀▀ ▄▀█   █▀▀ ▄█
    #░█░ █▀█ █▀▄ ██▄ █▀█   █▀░ ░█
    #x_train_tuberculosis = x_train_tuberculosis[:3]
    #y_train_tuberculosis = y_train_tuberculosis[:3]
    
    #█▀█ █░█ █▀▀ █▀█ █▀ ▄▀█ █▀▄▀█ █▀█ █░░ █▀▀
    #█▄█ ▀▄▀ ██▄ █▀▄ ▄█ █▀█ █░▀░█ █▀▀ █▄▄ ██▄ 
    '''
    num_samples_minority = len(x_train_tuberculosis)
    additional_samples_needed = len(x_train_normal) - num_samples_minority
    
    num_repeats = additional_samples_needed // num_samples_minority
    remaining_samples = additional_samples_needed % num_samples_minority
    
    x_train_tuberculosis_resampled = np.tile(x_train_tuberculosis, (num_repeats, 1, 1, 1))
    y_train_tuberculosis_resampled = np.tile(y_train_tuberculosis, (num_repeats))
    
    x_train_tuberculosis = np.concatenate((x_train_tuberculosis_resampled,x_train_tuberculosis, x_train_tuberculosis[:remaining_samples]), axis=0)
    y_train_tuberculosis = np.concatenate((y_train_tuberculosis_resampled,y_train_tuberculosis, y_train_tuberculosis[:remaining_samples]), axis=0)
    '''
    if dimensiones == 1:
        visualizacion.mostrarDimensiones(x_train_normal,y_train_normal,x_test_normal,x_train_tuberculosis,y_train_tuberculosis,x_test_tuberculosis)

    # Concatenar los conjuntos de evaluación de ambas clases
    x_test = np.concatenate((x_test_normal, x_test_tuberculosis), axis=0)
    y_test = np.concatenate((y_test_normal, y_test_tuberculosis), axis=0)

    # Concatenar los conjuntos de entrenamiento de ambas clases
    x_train = np.concatenate((x_train_normal, x_train_tuberculosis), axis=0)
    y_train = np.concatenate((y_train_normal, y_train_tuberculosis), axis=0)
   
    
    #█▀▀ █░░ ▄▀█ █▀ █▀   █░█░█ █▀▀ █ █▀▀ █░█ ▀█▀
    #█▄▄ █▄▄ █▀█ ▄█ ▄█   ▀▄▀▄▀ ██▄ █ █▄█ █▀█ ░█░
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(weights))
    
    if dimensiones == 1:
        print(f'Tamaño del conjunto de entrenamiento: {len(x_train)} & Tamaño del conjunto de test: {len(x_test)}')   
    
    return x_train, y_train, x_test, y_test,class_weights


