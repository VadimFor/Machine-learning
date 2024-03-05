import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

import numpy as np

# Si disponemos de GPU, la usamos
dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#█▀█ ▄▀█ █▀ █▀█   ▄█ ▀   █▀█ █▄▄ ▀█▀ █▀▀ █▄░█ █▀▀ █▀█   █▄█   ▀█▀ █▀█ ▄▀█ █▄░█ █▀ █▀▀ █▀█ █▀█ █▀▄▀█ ▄▀█ █▀█
#█▀▀ █▀█ ▄█ █▄█   ░█ ▄   █▄█ █▄█ ░█░ ██▄ █░▀█ ██▄ █▀▄   ░█░   ░█░ █▀▄ █▀█ █░▀█ ▄█ █▀░ █▄█ █▀▄ █░▀░█ █▀█ █▀▄

#█▀▄ ▄▀█ ▀█▀ █▀█ █▀
#█▄▀ █▀█ ░█░ █▄█ ▄█

tam_batch = 256

# Datos de entrenamiento y validación
trainval_dataset = datasets.CIFAR10(
    root='./datos', # Ruta donde se almacenarán los datos descargados
    train=True,     # Datos de entrenamiento (y validación)
    transform=transforms.ToTensor(), # Conversión más normalización
    download=True   # Se descargan a la carpeta indicada en root
)

# Dividimos los datos de entrenamiento en entrenamiento y validación
train_tam = int(0.8 * len(trainval_dataset))    # 80% para entrenamiento
val_tam = len(trainval_dataset) - train_tam     # 20% para validación
train_dataset, val_dataset = random_split(trainval_dataset,\
                                          [train_tam, val_tam])

# Datos de test
test_dataset = datasets.CIFAR10(
    root='./datos', # Misma ruta que los datos de entrenamiento
    train=False,    # Datos de test
    transform=transforms.ToTensor() # Conversión más normalización
)

# Creamos los dataloaders
train_loader = DataLoader(
    train_dataset,  # Dataset sobre el que iterar
    batch_size=tam_batch,
    shuffle=True    # Se mezclan los datos en cada época de entrenamiento
)
val_loader = DataLoader(
    val_dataset,    # Dataset sobre el que iterar
    batch_size=tam_batch,
    shuffle=False   # Los datos se evalúan siempre en el mismo orden
)
test_loader = DataLoader(
    test_dataset,
    batch_size=tam_batch,
    shuffle=False   # Los datos se evalúan siempre en el mismo orden
)


'''ＳＡＬＩＤＡ ＣＯＮＳＯＬＡ:
    Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./datos/cifar-10-python.tar.gz
    100%|██████████| 170498071/170498071 [00:23<00:00, 7350557.85it/s] 
    Extracting ./datos/cifar-10-python.tar.gz to ./datos
'''


# Dimensiones de los conjuntos de datos
print(f'Tamaño del conjunto de entrenamiento: {len(train_dataset)}')
print(f'Tamaño del conjunto de validación: {len(val_dataset)}')
print(f'Tamaño del conjunto de test: {len(test_dataset)}')

# Obtenemos un lote o batch de imágenes de entrenamiento
imagenes, etiquetas = next(iter(train_loader))
print(f'Tamaño de las imágenes: {imagenes.shape[1:]}')

# Visualizamos las imágenes y etiquetas de un batch de entrenamiento
fig, axs = plt.subplots(1, 2, figsize=(10, 10))
for i in range(2):
    axs[i].imshow(imagenes[i].permute(1, 2, 0))
    axs[i].set_title(f'Etiqueta: \
                     {trainval_dataset.classes[etiquetas[i]]}')
    axs[i].axis('off')
plt.show()

'''ＳＡＬＩＤＡ ＣＯＮＳＯＬＡ:
    Tamaño del conjunto de entrenamiento: 40000
    Tamaño del conjunto de validación: 10000
    Tamaño del conjunto de test: 10000
    Tamaño de las imágenes: torch.Size([3, 32, 32])
    [Nota: Saldrán 2 imagenes con sus etiquetas]
'''

#█▀█ ▄▀█ █▀ █▀█   ▀█ ▀   █▀▄ █▀▀ █▀▀ █ █▄░█ █ █▀█   ▄▀█ █▀█ █▀█ █░█ █ ▀█▀ █▀▀ █▀▀ ▀█▀ █░█ █▀█ ▄▀█
#█▀▀ █▀█ ▄█ █▄█   █▄ ▄   █▄▀ ██▄ █▀░ █ █░▀█ █ █▀▄   █▀█ █▀▄ ▀▀█ █▄█ █ ░█░ ██▄ █▄▄ ░█░ █▄█ █▀▄ █▀█

class CNN(nn.Module):
    def __init__(self, normalizacion=False, dropout=0):
        super(CNN, self).__init__()

        # 1) Bloque extractor de características
        # - Convolución con activación -> Normalización -> Pooling
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding='same')
        if normalizacion:
            self.norm1 = nn.BatchNorm2d(16) # Núm. Canales
        else:
            self.norm1 = nn.Identity()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # - Convolución con activación -> Normalización -> Pooling
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding='same')
        if normalizacion:
            self.norm2 = nn.BatchNorm2d(32) # Núm. Canales
        else:
            self.norm2 = nn.Identity()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        # - Dropout
        self.dropout = nn.Dropout(dropout)

        # 2) Bloque clasificador de características
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(32 * 8 * 8, 256)
        self.lin2 = nn.Linear(256, 128)
        self.lin3 = nn.Linear(128, 10)

        # Activaciones:
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.norm1(x)
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.norm2(x)
        x = self.pool2(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.relu(self.lin1(x))
        x = self.relu(self.lin2(x))
        y = self.lin3(x)
        return y


#█▀█ ▄▀█ █▀ █▀█  ▀█ ▀   █▀▀ █░█ █▄░█ █▀▀ █ █▀█ █▄░█ █▀▀ █▀   █▀▄ █▀▀ █░░   █▀▄▀█ █▀█ █▀▄ █▀▀ █░░ █▀█
#█▀▀ █▀█ ▄█ █▄█  ▄█ ▄   █▀░ █▄█ █░▀█ █▄▄ █ █▄█ █░▀█ ██▄ ▄█   █▄▀ ██▄ █▄▄   █░▀░█ █▄█ █▄▀ ██▄ █▄▄ █▄█

def epoca_train(modelo, train_loader, optimizador,\
                func_perdida, dispositivo, aumentos=None):
    epoca_perdidas = []
    # Iteramos sobre los datos de entrenamiento
    for i, (imagenes, etiquetas) in enumerate(train_loader):
        # Si es posible, transferimos los datos a la GPU
        imagenes = imagenes.to(dispositivo)
        etiquetas = etiquetas.to(dispositivo)
        # Aumento de datos
        if aumentos is not None:
            imagenes = aumentos(imagenes)
        # Ponemos los gradientes a cero
        optimizador.zero_grad()
        # Calculamos las salidas de la red neuronal
        # para las imágenes de entrada
        salidas = modelo(imagenes)
        # Calculamos la pérdida de la red neuronal
        perdida = func_perdida(salidas, etiquetas)
        # Calculamos los gradientes de la pérdida
        # respecto a los parámetros de la red neuronal
        perdida.backward()
        # Actualizamos los parámetros de la red neuronal
        # usando el algoritmo de optimización
        optimizador.step()
        # Añadimos a la lista de pérdidas la pérdida actual
        epoca_perdidas.append(perdida.item())
    # Calculamos la media de las pérdidas para una época
    return sum(epoca_perdidas) / len(epoca_perdidas)

def epoca_test(modelo, test_loader, dispositivo):
    # Establecemos la red neuronal en modo de evaluación
    modelo.eval()
    # Desactivamos el cálculo de gradientes
    with torch.no_grad():
        correcto = 0
        total = 0
        # Iteramos sobre los datos de test
        for imagenes, etiquetas in test_loader:
            # Si es posible, transferimos los datos a la GPU
            imagenes = imagenes.to(dispositivo)
            etiquetas = etiquetas.to(dispositivo)
            # Calculamos las salidas de la red neuronal
            # para las imágenes de entrada
            salidas = modelo(imagenes)
            # Calculamos las probabilidades de las clases
            salidas = salidas.softmax(dim=1)
            # Obtenemos la clase con mayor probabilidad
            prediccion = salidas.argmax(dim=1)
            # Incrementamos el contador de imágenes procesadas
            total += etiquetas.size(0)
            # Incrementamos el contador de imágenes
            # clasificadas correctamente
            correcto += (prediccion == etiquetas).sum().item()
        # Calculamos la exactitud
        exactitud = 100 * correcto / total
    # Establecemos la red neuronal en modo entrenamiento
    modelo.train()
    return exactitud


#█▀█ ▄▀█ █▀ █▀█   █░█ ▀   █▀▀ █▀█ █▀▄▀█ █▀█ █ █░░ ▄▀█ █▀█   █▄█   █▀▀ █▄░█ ▀█▀ █▀█ █▀▀ █▄░█ ▄▀█ █▀█
#█▀▀ █▀█ ▄█ █▄█   ▀▀█ ▄   █▄▄ █▄█ █░▀░█ █▀▀ █ █▄▄ █▀█ █▀▄   ░█░   ██▄ █░▀█ ░█░ █▀▄ ██▄ █░▀█ █▀█ █▀▄

#█▀▄▀█ █▀█ █▀▄ █▀▀ █░░ █▀█
#█░▀░█ █▄█ █▄▀ ██▄ █▄▄ █▄█

def entrenar(modelo, train_loader, val_loader, test_loader,\
             epocas, optimizador, func_perdida, dispositivo,\
             early_stop=False, paciencia=0, aumentos=None):
    # Históricos de valores de pérdida de entrenamiento
    train_perdidas = []
    # Históricos de valores de exactitud de entrenamiento,
    # validación y test
    train_accs, val_accs, test_accs = [], [], []
    # Mejor exactitud de validación
    mejor_val_acc = 0
    # Contador de épocas desde la última mejora en validación
    epocas_desde_mejora = 0

    # Iteramos por cada época
    for epoca in range(epocas):
        # Entrenamos la red neuronal por una época
        train_perdida = epoca_train(modelo, train_loader, optimizador,\
                                    func_perdida, dispositivo, aumentos)
        train_perdidas.append(train_perdida)
        # Calculamos la exactitud para los datos de entrenamiento
        train_acc = epoca_test(modelo, train_loader, dispositivo)
        train_accs.append(train_acc)
        # Calculamos la exactitud para los datos de validación
        valid_acc = epoca_test(modelo, val_loader, dispositivo)
        val_accs.append(valid_acc)
        # Calculamos la exactitud para los datos de test
        test_acc = epoca_test(modelo, test_loader, dispositivo)
        test_accs.append(test_acc)
        # Sacamos los resultados por pantalla
        print(f'Época {epoca+1}/{epocas},\
        pérdida={train_perdida:.4f},\
        exactitud_train={train_acc:.2f}%,\
        exactitud_val={valid_acc:.2f}%,\
        exactitud_test={test_acc:.2f}%')

        # Early stopping
        if early_stop:
            if valid_acc > mejor_val_acc:
                mejor_val_acc = valid_acc
                epocas_desde_mejora = 0  # Reset
            else:
                epocas_desde_mejora += 1
            # Si la exactitud de validación no mejora tras "paciencia"
            # épocas, paramos el entrenamiento
            if epocas_desde_mejora == paciencia:
                print(f'No se ha mejorado la tasa de acierto en\
                      validación en {paciencia} épocas.')
                break  # Salimos del bucle de las épocas

    # Se ha completado el entrenamiento de la red neuronal
    print('Fin del entrenamiento.')
    return train_perdidas, train_accs, val_accs, test_accs

#Definición de los hiperparámetros comúnes a ambos escenarios
epocas = 50
func_perdida = nn.CrossEntropyLoss()


#ENTRENAMIENTO DEL CASO BASE
# Caso base
modelo_base = CNN().to(dispositivo)
optim_base = optim.Adam(modelo_base.parameters())

# Entrenamos el modelo base
metricas_base = entrenar(modelo_base, train_loader, val_loader,\
                         test_loader, epocas, optim_base,\
                         func_perdida, dispositivo)

'''ＳＡＬＩＤＡ ＣＯＮＳＯＬＡ:
   Época 1/50,        pérdida=1.8339,        exactitud_train=42.76%,        exactitud_val=43.40%,        exactitud_test=43.01%
    Época 2/50,        pérdida=1.5053,        exactitud_train=49.36%,        exactitud_val=49.24%,        exactitud_test=49.34%
    Época 3/50,        pérdida=1.3795,        exactitud_train=53.93%,        exactitud_val=53.06%,        exactitud_test=53.07%
    Época 4/50,        pérdida=1.2773,        exactitud_train=57.12%,        exactitud_val=55.78%,        exactitud_test=55.25%
    Época 5/50,        pérdida=1.1870,        exactitud_train=60.42%,        exactitud_val=57.68%,        exactitud_test=57.62%
    Época 6/50,        pérdida=1.1096,        exactitud_train=61.84%,        exactitud_val=58.66%,        exactitud_test=58.70%
    Época 7/50,        pérdida=1.0571,        exactitud_train=63.87%,        exactitud_val=60.57%,        exactitud_test=60.35%
    Época 8/50,        pérdida=0.9879,        exactitud_train=65.26%,        exactitud_val=61.73%,        exactitud_test=60.74%
    Época 9/50,        pérdida=0.9466,        exactitud_train=67.45%,        exactitud_val=62.21%,        exactitud_test=61.72%
    Época 10/50,        pérdida=0.8909,        exactitud_train=70.85%,        exactitud_val=64.16%,        exactitud_test=64.38%
    Época 11/50,        pérdida=0.8351,        exactitud_train=70.51%,        exactitud_val=62.71%,        exactitud_test=62.73%
    Época 12/50,        pérdida=0.7903,        exactitud_train=74.70%,        exactitud_val=65.58%,        exactitud_test=65.86%
    Época 13/50,        pérdida=0.7385,        exactitud_train=75.58%,        exactitud_val=65.84%,        exactitud_test=64.88%
    Época 14/50,        pérdida=0.6936,        exactitud_train=78.17%,        exactitud_val=66.58%,        exactitud_test=66.66%
    Época 15/50,        pérdida=0.6594,        exactitud_train=78.64%,        exactitud_val=66.88%,        exactitud_test=66.11%
    Época 16/50,        pérdida=0.6047,        exactitud_train=82.03%,        exactitud_val=67.49%,        exactitud_test=67.47%
    Época 17/50,        pérdida=0.5580,        exactitud_train=83.23%,        exactitud_val=67.03%,        exactitud_test=66.48%
    Época 18/50,        pérdida=0.5099,        exactitud_train=83.31%,        exactitud_val=66.80%,        exactitud_test=66.31%
    Época 19/50,        pérdida=0.4715,        exactitud_train=85.73%,        exactitud_val=67.22%,        exactitud_test=66.86%
    Época 20/50,        pérdida=0.4192,        exactitud_train=87.44%,        exactitud_val=66.73%,        exactitud_test=66.64%
    Época 21/50,        pérdida=0.3852,        exactitud_train=88.55%,        exactitud_val=66.61%,        exactitud_test=66.73%
    Época 22/50,        pérdida=0.3450,        exactitud_train=91.10%,        exactitud_val=67.67%,        exactitud_test=66.60%
    Época 23/50,        pérdida=0.3089,        exactitud_train=89.77%,        exactitud_val=65.82%,        exactitud_test=65.18%
    Época 24/50,        pérdida=0.2778,        exactitud_train=92.42%,        exactitud_val=66.76%,        exactitud_test=66.87%
    Época 25/50,        pérdida=0.2363,        exactitud_train=92.29%,        exactitud_val=66.84%,        exactitud_test=65.96%
    ...
    Época 48/50,        pérdida=0.0586,        exactitud_train=98.01%,        exactitud_val=65.16%,        exactitud_test=64.95%
    Época 49/50,        pérdida=0.0623,        exactitud_train=98.36%,        exactitud_val=66.05%,        exactitud_test=65.13%
    Época 50/50,        pérdida=0.0541,        exactitud_train=98.81%,        exactitud_val=65.59%,        exactitud_test=65.36%
    Fin del entrenamiento.
'''

#######################################CASO AVANZADO###############
aumentos = transforms.Compose([
    transforms.RandomRotation(5),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomApply(nn.ModuleList(
        [transforms.ColorJitter(brightness=0.2, hue=0.2)]), p=0.1)])

# Visualizamos las imágenes aumentadas de un batch de entrenamiento
fig, axs = plt.subplots(1, 2, figsize=(10, 10))
for i in range(2):
    axs[i].imshow(aumentos(imagenes[i]).permute(1, 2, 0))
    axs[i].axis('off')
plt.show()


'''ＳＡＬＩＤＡ ＣＯＮＳＯＬＡ:
    [Nota : Saldrá ventana con 2 imagenes distorsionadas]
'''
#ENTRENAMIENTO CASO AVANZADO
# Caso avanzado
modelo_avan = CNN(normalizacion=True, dropout=0.5).to(dispositivo)
optim_avan = optim.Adam(modelo_avan.parameters())

# Entrenamos el modelo avanzado
metricas_avan = entrenar(modelo_avan, train_loader, val_loader,\
                         test_loader, epocas, optim_avan,\
                         func_perdida, dispositivo, early_stop=True,\
                         paciencia=5, aumentos=aumentos)

'''ＳＡＬＩＤＡ ＣＯＮＳＯＬＡ:
    Época 1/50,        pérdida=1.4728,        exactitud_train=58.10%,        exactitud_val=56.67%,        exactitud_test=56.18%
    Época 2/50,        pérdida=1.1902,        exactitud_train=62.70%,        exactitud_val=60.67%,        exactitud_test=60.32%
    Época 3/50,        pérdida=1.0648,        exactitud_train=57.30%,        exactitud_val=55.56%,        exactitud_test=55.31%
    Época 4/50,        pérdida=0.9745,        exactitud_train=66.16%,        exactitud_val=63.12%,        exactitud_test=62.90%
    Época 5/50,        pérdida=0.9208,        exactitud_train=70.15%,        exactitud_val=67.12%,        exactitud_test=66.49%
    Época 6/50,        pérdida=0.8580,        exactitud_train=74.01%,        exactitud_val=69.79%,        exactitud_test=68.96%
    Época 7/50,        pérdida=0.8181,        exactitud_train=75.16%,        exactitud_val=69.74%,        exactitud_test=69.50%
    Época 8/50,        pérdida=0.7850,        exactitud_train=79.08%,        exactitud_val=72.35%,        exactitud_test=72.04%
    Época 9/50,        pérdida=0.7515,        exactitud_train=69.21%,        exactitud_val=63.69%,        exactitud_test=63.25%
    Época 10/50,        pérdida=0.7258,        exactitud_train=79.37%,        exactitud_val=71.72%,        exactitud_test=71.67%
    Época 11/50,        pérdida=0.7006,        exactitud_train=82.24%,        exactitud_val=74.06%,        exactitud_test=73.82%
    Época 12/50,        pérdida=0.6713,        exactitud_train=82.98%,        exactitud_val=74.16%,        exactitud_test=74.17%
    Época 13/50,        pérdida=0.6480,        exactitud_train=80.44%,        exactitud_val=72.09%,        exactitud_test=71.44%
    Época 14/50,        pérdida=0.6417,        exactitud_train=81.22%,        exactitud_val=72.15%,        exactitud_test=72.29%
    Época 15/50,        pérdida=0.6145,        exactitud_train=85.12%,        exactitud_val=74.99%,        exactitud_test=74.40%
    Época 16/50,        pérdida=0.6092,        exactitud_train=81.77%,        exactitud_val=72.47%,        exactitud_test=72.31%
    Época 17/50,        pérdida=0.5898,        exactitud_train=69.55%,        exactitud_val=63.09%,        exactitud_test=62.25%
    Época 18/50,        pérdida=0.5723,        exactitud_train=84.01%,        exactitud_val=73.41%,        exactitud_test=72.38%
    Época 19/50,        pérdida=0.5675,        exactitud_train=82.86%,        exactitud_val=72.25%,        exactitud_test=72.00%
    Época 20/50,        pérdida=0.5463,        exactitud_train=83.94%,        exactitud_val=73.43%,        exactitud_test=72.81%
    No se ha mejorado la tasa de acierto en                      validación en 5 épocas.
    Fin del entrenamiento.
'''

#█▀█ ▄▀█ █▀ █▀█   █▀ ▀   █▀█ █▀▀ █▀ █░█ █░░ ▀█▀ ▄▀█ █▀▄ █▀█ █▀   █▄█   █▀█ █▀█ █▀▀ █▀▄ █ █▀▀ █▀▀ █ █▀█ █▄░█
#█▀▀ █▀█ ▄█ █▄█   ▄█ ▄   █▀▄ ██▄ ▄█ █▄█ █▄▄ ░█░ █▀█ █▄▀ █▄█ ▄█   ░█░   █▀▀ █▀▄ ██▄ █▄▀ █ █▄▄ █▄▄ █ █▄█ █░▀█


# Calcular la exactitud de test en función de la mejor exactitud de validación
def test_acc_from_best_val_acc(metricas):
    # Obtenemos la época asociada a la mejor exactitud de validación
    epoca = np.argmax(metricas[2])
    # Devolvemos la exactitud de test de esa época
    return metricas[3][epoca]

# Exactitud de test del modelo base
test_acc_base = test_acc_from_best_val_acc(metricas_base)
print(f'Exactitud de test del modelo base: {test_acc_base:.2f}%')

# Exactitud de test del modelo avanzado
test_acc_avan = test_acc_from_best_val_acc(metricas_avan)
print(f'Exactitud de test del modelo avanzado: {test_acc_avan:.2f}%')

# Visualizamos la perdida de entrenamiento y la exactitud de test
# a lo largo de las épocas para ambos modelos
fig, axs = plt.subplots(1, 2, figsize=(15, 5))
res_comp = {'Pérdida de entrenamiento': 0, 'Exactitud de test': -1}
for i, (k, v) in enumerate(res_comp.items()):
    axs[i].plot(metricas_base[v], label='Base')
    axs[i].plot(metricas_avan[v], label='Avanzado')
    axs[i].set_xlabel('Época')
    axs[i].set_ylabel(k)
    axs[i].legend()
plt.show()

'''ＳＡＬＩＤＡ ＣＯＮＳＯＬＡ:
    Exactitud de test del modelo base: 66.60%
    Exactitud de test del modelo avanzado: 74.40%
    [Nota: Saldrá 2 gráficas de comparación entre base y avanzado]
'''

