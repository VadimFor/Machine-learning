import datos
import visualizacion
import auxiliares
from tensorflow.keras import layers, models
import time

import tensorflow as tf

#█▀█ ▄▀█ █▀ █▀█   ▄█ ▀   █▀█ █▄▄ ▀█▀ █▀▀ █▄░█ █▀▀ █▀█   █▄█   ▀█▀ █▀█ ▄▀█ █▄░█ █▀ █▀▀ █▀█ █▀█ █▀▄▀█ ▄▀█ █▀█
#█▀▀ █▀█ ▄█ █▄█   ░█ ▄   █▄█ █▄█ ░█░ ██▄ █░▀█ ██▄ █▀▄   ░█░   ░█░ █▀▄ █▀█ █░▀█ ▄█ █▀░ █▄█ █▀▄ █░▀░█ █▀█ █▀▄

#█▀▄ ▄▀█ ▀█▀ █▀█ █▀
#█▄▀ █▀█ ░█░ █▄█ ▄█

x_train, y_train, x_test, y_test = datos.obtener_train_test(1)
#datos.mostrar_proporciones_clases(x_train_normal, x_train_tuberculosis)
#datos.mostrar_imagen_etiqueta(x_train,y_train)

#█▀█ ▄▀█ █▀ █▀█   ▀█ ▀   █▀▄ █▀▀ █▀▀ █ █▄░█ █ █▀█   ▄▀█ █▀█ █▀█ █░█ █ ▀█▀ █▀▀ █▀▀ ▀█▀ █░█ █▀█ ▄▀█
#█▀▀ █▀█ ▄█ █▄█   █▄ ▄   █▄▀ ██▄ █▀░ █ █░▀█ █ █▀▄   █▀█ █▀▄ ▀▀█ █▄█ █ ░█░ ██▄ █▄▄ ░█░ █▄█ █▀▄ █▀█

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
])'''

#█▀█ ▄▀█ █▀ █▀█  ▀█ ▀   █▀▀ █▄░█ ▀█▀ █▀█ █▀▀ █▄░█ ▄▀█ █▀█   █▀▄▀█ █▀█ █▀▄ █▀▀ █░░ █▀█
#█▀▀ █▀█ ▄█ █▄█  ▄█ ▄   ██▄ █░▀█ ░█░ █▀▄ ██▄ █░▀█ █▀█ █▀▄   █░▀░█ █▄█ █▄▀ ██▄ █▄▄ █▄█

# Compilar el modelo
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# MOSTRAR SUMMARY
model.summary()

start_time = time.time()
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))
end_time = time.time()

elapsed_time = (end_time - start_time) / 60 
print(f"Entrenamiento completado en {elapsed_time:.2f} minutos.")

## GUARDAR HISTORY ##
auxiliares.guardar_history(history)


#█▀█ ▄▀█ █▀ █▀█   █░█ ▀   █▀█ █▀▀ █▀ █░█ █░░ ▀█▀ ▄▀█ █▀▄ █▀█ █▀   █▄█   █▀█ █▀█ █▀▀ █▀▄ █ █▀▀ █▀▀ █ █▀█ █▄░█
#█▀▀ █▀█ ▄█ █▄█   ▀▀█ ▄   █▀▄ ██▄ ▄█ █▄█ █▄▄ ░█░ █▀█ █▄▀ █▄█ ▄█   ░█░   █▀▀ █▀▄ ██▄ █▄▀ █ █▄▄ █▄▄ █ █▄█ █░▀█

visualizacion.mostrar_grafico_accuracy(history)
visualizacion.mostrar_grafico_loss(history)

#█▀▀ ▄█
#█▀░ ░█
visualizacion.mostrar_f1(model,x_test,y_test)

#█▀▀ █░█ ▄▀█ █▀█ █▀▄ ▄▀█ █▀█   █▀▄▀█ █▀█ █▀▄ █▀▀ █░░ █▀█
#█▄█ █▄█ █▀█ █▀▄ █▄▀ █▀█ █▀▄   █░▀░█ █▄█ █▄▀ ██▄ █▄▄ █▄█

model.save('tb_modelo_2')

#█▀▀ ▄▀█ █▀█ █▀▀ ▄▀█ █▀█   █▀▄▀█ █▀█ █▀▄ █▀▀ █░░ █▀█
#█▄▄ █▀█ █▀▄ █▄█ █▀█ █▀▄   █░▀░█ █▄█ █▄▀ ██▄ █▄▄ █▄█


