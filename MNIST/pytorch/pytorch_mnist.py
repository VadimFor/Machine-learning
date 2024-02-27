
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch import optim

#█▀█ ▄▀█ █▀ █▀█   ▄█ ▀   █▀█ █▄▄ ▀█▀ █▀▀ █▄░█ █▀▀ █▀█   █▄█   ▀█▀ █▀█ ▄▀█ █▄░█ █▀ █▀▀ █▀█ █▀█ █▀▄▀█ ▄▀█ █▀█
#█▀▀ █▀█ ▄█ █▄█   ░█ ▄   █▄█ █▄█ ░█░ ██▄ █░▀█ ██▄ █▀▄   ░█░   ░█░ █▀▄ █▀█ █░▀█ ▄█ █▀░ █▄█ █▀▄ █░▀░█ █▀█ █▀▄

#█▀▄ ▄▀█ ▀█▀ █▀█ █▀
#█▄▀ █▀█ ░█░ █▄█ ▄█
tam_batch = 32

# Datos de entrenamiento:
train_dataset = datasets.MNIST(
    root='./datos', # Ruta donde se almacenarán los datos descargados
    train=True,     # Datos de entrenamiento
    transform=transforms.ToTensor(), # Conversión más normalización
    download=True   # Se descargan a la carpeta indicada en root
)

train_set, val_set = torch.utils.data.random_split(train_dataset, [50000, 10000])

train_loader = torch.utils.data.DataLoader(
    train_set,      # Dataset sobre el que iterar
    batch_size=tam_batch,
    shuffle=True    # Se mezclan los datos en cada época de entrenamiento
)

val_loader = torch.utils.data.DataLoader(
    val_set,        # Dataset sobre el que iterar
    batch_size=tam_batch,
    shuffle=True    # Se mezclan los datos en cada época de entrenamiento
)

# Datos de test
test_dataset = datasets.MNIST(
    root='./datos', # Misma ruta que los datos de entrenamiento
    train=False,    # Datos de test
    transform=transforms.ToTensor() # Conversión más normalización
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=tam_batch,
    shuffle=False   # Los datos se evalúan siempre en el mismo orden
)

# Dimensiones de los conjuntos de datos
print(f'Tamaño del conjunto de entrenamiento: {len(train_set)}')
print(f'Tamaño del conjunto de validación: {len(val_set)}')
print(f'Tamaño del conjunto de test: {len(test_dataset)}')

# Obtenemos un lote o batch de imágenes de entrenamiento
imagenes, etiquetas = next(iter(train_loader))
print(f'Tamaño de las imágenes: {imagenes.shape[1:]}')

# Visualizamos un imagen junto a su etiqueta
id_ejemplo = 10
plt.imshow(np.squeeze(imagenes[id_ejemplo]), cmap='gray')
plt.show()
print(f'Ejemplo de etiqueta: {etiquetas[id_ejemplo]}')

'''ＳＡＬＩＤＡ ＣＯＮＳＯＬＡ:
    tamaño del conjunto de entrenamiento: 50000
    Tamaño del conjunto de validación: 10000
    Tamaño del conjunto de test: 10000
    Tamaño de las imágenes: torch.Size([1, 28, 28])
    
    [Nota: Saldrá una ventana nueva con la imagen seleccionada]
    Ejemplo de etiqueta: [Nota: saldrá diferente imagen con cada ejecución]
'''

#█▀█ ▄▀█ █▀ █▀█   ▀█ ▀   █▀▄ █▀▀ █▀▀ █ █▄░█ █ █▀█   ▄▀█ █▀█ █▀█ █░█ █ ▀█▀ █▀▀ █▀▀ ▀█▀ █░█ █▀█ ▄▀█
#█▀▀ █▀█ ▄█ █▄█   █▄ ▄   █▄▀ ██▄ █▀░ █ █░▀█ █ █▀▄   █▀█ █▀▄ ▀▀█ █▄█ █ ░█░ ██▄ █▄▄ ░█░ █▄█ █▀▄ █▀█

class MLP(nn.Module):
  def __init__(self):
    super(MLP, self).__init__()

    self.backbone = nn.Sequential(
        nn.Linear(in_features=784, out_features=512),
        nn.ReLU(),
        nn.Linear(in_features=512, out_features=10)
    )

  def forward(self, input):
    input = torch.flatten(input.squeeze(1), start_dim=1, end_dim=2)
    return self.backbone(input)


#█▀█ ▄▀█ █▀ █▀█   ▀█ ▀   █▀▀ █▀█ █▀▀ ▄▀█ █▀█   █▀▄▀█ █▀█ █▀▄ █▀▀ █░░ █▀█
#█▀▀ █▀█ ▄█ █▄█   ▄█ ▄   █▄▄ █▀▄ ██▄ █▀█ █▀▄   █░▀░█ █▄█ █▄▀ ██▄ █▄▄ █▄█

modelo = MLP()
print(modelo)

# Si disponemos de GPU, la usamos
dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
modelo.to(dispositivo)
print(f'Usando {dispositivo}')

func_perdida = nn.CrossEntropyLoss()
optimizador = optim.Adam(modelo.parameters())

'''ＳＡＬＩＤＡ ＣＯＮＳＯＬＡ:
    MLP(
    (backbone): Sequential(
        (0): Linear(in_features=784, out_features=512, bias=True)
        (1): ReLU()
        (2): Linear(in_features=512, out_features=10, bias=True)
    )
    )
    Usando cpu
'''

#█▀█ ▄▀█ █▀ █▀█   █░█ ▀   █▀▀ █░█ █▄░█ █▀▀ █ █▀█ █▄░█ █▀▀ █▀   █▀▄ █▀▀ █░░   █▀▄▀█ █▀█ █▀▄ █▀▀ █░░ █▀█
#█▀▀ █▀█ ▄█ █▄█   ▀▀█ ▄   █▀░ █▄█ █░▀█ █▄▄ █ █▄█ █░▀█ ██▄ ▄█   █▄▀ ██▄ █▄▄   █░▀░█ █▄█ █▄▀ ██▄ █▄▄ █▄█

#Ｆｕｎｃｉｏｎ  ｄｅ  ｅｎｔｒｅｎａｍｉｅｎｔｏ  ｄｅｌ  ｍｏｄｅｌｏ
def epoca_train(modelo, train_loader, optimizador, func_perdida, dispositivo):
    epoca_perdidas = []

    # Iteramos sobre los datos de entrenamiento
    for i, (imagenes, etiquetas) in enumerate(train_loader):
        # Si es posible, transferimos los datos a la GPU
        imagenes = imagenes.to(dispositivo)
        etiquetas = etiquetas.to(dispositivo)

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

#Ｆｕｎｃｉｏｎ  ｄｅ  ｅｖａｌｕａｃｉｏｎ  ｄｅｌ  ｍｏｄｅｌｏ
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


#█▀█ ▄▀█ █▀ █▀█   █▀ ▀   █▀▀ █▄░█ ▀█▀ █▀█ █▀▀ █▄░█ ▄▀█ █▀█   █▀▄▀█ █▀█ █▀▄ █▀▀ █░░ █▀█
#█▀▀ █▀█ ▄█ █▄█   ▄█ ▄   ██▄ █░▀█ ░█░ █▀▄ ██▄ █░▀█ █▀█ █▀▄   █░▀░█ █▄█ █▄▀ ██▄ █▄▄ █▄█

train_accs = [] # Histórico de valores de exactitud de entrenamiento
val_accs = [] # Histórico de valores de exactitud de test
epocas = 15 # Definimos el número de épocas de entrenamiento

# Iteramos por cada época
for epoca in range(epocas):
    # Entrenamos la red neuronal por una época
    train_perdida = epoca_train(
        modelo, train_loader, optimizador, func_perdida, dispositivo)

    # Calculamos la exactitud para los datos de entrenamiento
    train_acc = epoca_test(modelo, train_loader, dispositivo)
    train_accs.append(train_acc)

    # Calculamos la exactitud para los datos de test
    val_acc = epoca_test(modelo, val_loader, dispositivo)
    val_accs.append(val_acc)

    # Sacamos los resultados por pantalla
    print(f'Época {epoca+1}/{epocas},\
            pérdida={train_perdida:.4f},\
            exactitud_train={train_acc:.2f}%,\
            exactitud_validacion={val_acc:.2f}%')

# Se ha completado el entrenamiento de la red neuronal
print('Fin del entrenamiento.')

'''ＳＡＬＩＤＡ ＣＯＮＳＯＬＡ:
    Usando cpu
    Época 1/15,            pérdida=0.2435,            exactitud_train=97.02%,            exactitud_validacion=96.27%
    Época 2/15,            pérdida=0.0950,            exactitud_train=98.28%,            exactitud_validacion=97.11%
    Época 3/15,            pérdida=0.0611,            exactitud_train=98.69%,            exactitud_validacion=97.57%
    Época 4/15,            pérdida=0.0426,            exactitud_train=99.38%,            exactitud_validacion=97.72%
    Época 5/15,            pérdida=0.0291,            exactitud_train=99.05%,            exactitud_validacion=97.60%
    Época 6/15,            pérdida=0.0230,            exactitud_train=99.52%,            exactitud_validacion=98.01%
    Época 7/15,            pérdida=0.0179,            exactitud_train=99.64%,            exactitud_validacion=97.90%
    Época 8/15,            pérdida=0.0139,            exactitud_train=99.72%,            exactitud_validacion=97.96%
    Época 9/15,            pérdida=0.0121,            exactitud_train=99.79%,            exactitud_validacion=98.09%
    Época 10/15,            pérdida=0.0103,            exactitud_train=99.59%,            exactitud_validacion=97.71%
    Época 11/15,            pérdida=0.0103,            exactitud_train=99.71%,            exactitud_validacion=97.90%
    Época 12/15,            pérdida=0.0077,            exactitud_train=99.63%,            exactitud_validacion=97.77%
    Época 13/15,            pérdida=0.0083,            exactitud_train=99.65%,            exactitud_validacion=97.60%
    Época 14/15,            pérdida=0.0073,            exactitud_train=99.94%,            exactitud_validacion=98.01%
    Época 15/15,            pérdida=0.0061,            exactitud_train=99.78%,            exactitud_validacion=97.78%
    Fin del entrenamiento.
'''

#█▀█ ▄▀█ █▀ █▀█   █▄▄ ▀   █▀█ █▀▀ █▀ █░█ █░░ ▀█▀ ▄▀█ █▀▄ █▀█ █▀
#█▀▀ █▀█ ▄█ █▄█   █▄█ ▄   █▀▄ ██▄ ▄█ █▄█ █▄▄ ░█░ █▀█ █▄▀ █▄█ ▄█

plt.plot(train_accs)
plt.plot(val_accs)
plt.xlabel('Época')         # Eje X - Época
plt.ylabel('Exactitud (%)') # Eje Y - Exactitud
plt.legend(['Entrenamiento', 'Validacion'])
plt.show()
'''ＳＡＬＩＤＡ ＣＯＮＳＯＬＡ:
[NOTA: Saldrá una ventana con un gráfico.]
'''

#█▀█ ▄▀█ █▀ █▀█   ▀▀█ ▀   █▀█ █▀█ █▀▀ █▀▄ █ █▀▀ █▀▀ █ █▀█ █▄░█
#█▀▀ █▀█ ▄█ █▄█   ░░█ ▄   █▀▀ █▀▄ ██▄ █▄▀ █ █▄▄ █▄▄ █ █▄█ █░▀█

test_acc = epoca_test(modelo, test_loader, dispositivo)
print(f"Exactitud test: {test_acc:.2f}%")

'''ＳＡＬＩＤＡ ＣＯＮＳＯＬＡ:
Exactitud test: 97.85%
'''

#█▀█ ▄▀█ █▀ █▀█   ▄▀▄ ▀   █▀▀ █░█ ▄▀█ █▀█ █▀▄ ▄▀█ █▀█   █▀▄▀█ █▀█ █▀▄ █▀▀ █░░ █▀█
#█▀▀ █▀█ ▄█ █▄█   ▀▄▀ ▄   █▄█ █▄█ █▀█ █▀▄ █▄▀ █▀█ █▀▄   █░▀░█ █▄█ █▄▀ ██▄ █▄▄ █▄█

torch.save(modelo.state_dict(), 'pytorch_mnist_pesos.pth')



#█▀█ ▄▀█ █▀ █▀█   ▄▀█ █▀▄ █ █▀▀ █ █▀█ █▄░█ ▄▀█ █░░ ▀   █▀▀ ▄▀█ █▀█ █▀▀ ▄▀█ █▀█   █▀▄▀█ █▀█ █▀▄ █▀▀ █░░ █▀█
#█▀▀ █▀█ ▄█ █▄█   █▀█ █▄▀ █ █▄▄ █ █▄█ █░▀█ █▀█ █▄▄ ▄   █▄▄ █▀█ █▀▄ █▄█ █▀█ █▀▄   █░▀░█ █▄█ █▄▀ ██▄ █▄▄ █▄█

nuevo_modelo = MLP()
nuevo_modelo.load_state_dict(torch.load('pytorch_mnist_pesos.pth'))
nuevo_modelo.to(dispositivo)

test_acc = epoca_test(nuevo_modelo, test_loader, dispositivo)

print(f"Exactitud test: {test_acc:.2f}%")