import datos
import visualizacion
import auxiliares
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import keras
import cv2  # Import cv2 for image processing
from tensorflow.keras.models import Model
from PIL import Image

LOAD_MODEL = False

if LOAD_MODEL==False:
    
    #█▀█ ▄▀█ █▀ █▀█   ▄█ ▀  █▀▄ ▄▀█ ▀█▀ █▀█ █▀
    #█▀▀ █▀█ ▄█ █▄█    █ ▄  █▄▀ █▀█  █  █▄█ ▄█
    
    x_train, y_train, x_test, y_test, class_weights = datos.obtener_train_test(1)
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
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    #model.add(layers.Dropout(0.5))  # Para evitar/reducir overfitting
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid')) 

    '''
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(512, 512, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])'''

    #█▀█ ▄▀█ █▀ █▀█  ▀█ ▀   █▀▀ █▄ █ ▀█▀ █▀█ █▀▀ █▄ █ ▄▀█ █▀█   █▀▄▀█ █▀█ █▀▄ █▀▀ █░░ █▀█
    #█▀▀ █▀█ ▄█ █▄█  ▄█ ▄   ██▄ █░▀█ ░█░ █▀▄ ██▄ █ ▀█ █▀█ █▀▄   █░▀░█ █▄█ █▄▀ ██▄ █▄▄ █▄█
    
    # Compilar el modelo
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

    # MOSTRAR SUMMARY
    model.summary()

    start_time = time.time()
    
    history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))
    
    #CLASS WEIGHTS
    #history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test), class_weight=class_weights)
    
    end_time = time.time()

    elapsed_time = (end_time - start_time) / 60 
    print(f"Entrenamiento completado en {elapsed_time:.2f} minutos.")
    
   
    #█▀▀ █░█ ▄▀█ █▀█ █▀▄ ▄▀█ █▀█   █▀▄▀█ █▀█ █▀▄ █▀▀ █░░ █▀█
    #█▄█ █▄█ █▀█ █▀▄ █▄▀ █▀█ █▀▄   █░▀░█ █▄█ █▄▀ ██▄ █▄▄ █▄█

    model.save('tb_modelo_basic_560')
    
    #█░█ █ █▀ ▀█▀ █▀█ █▀█ █▄█
    #█▀█ █ ▄█ ░█░ █▄█ █▀▄ ░█░

    #auxiliares.guardar_history(history) #ＧＵＡＲＤＡＲ

    #history = auxiliares.cargar_history #ＣＡＲＧＡＲ
    
    #█▀▀ ▄█
    #█▀░ ░█
    visualizacion.mostrar_f1(model,x_test,y_test)
    
    #█▀█ █▀▀ █▀ █░█ █░░ ▀█▀ ▄▀█ █▀▄ █▀█ █▀   █▄█   █▀█ █▀█ █▀▀ █▀▄ █ █▀▀ █▀▀ █ █▀█ █▄░█
    #█▀▄ ██▄ ▄█ █▄█ █▄▄ ░█░ █▀█ █▄▀ █▄█ ▄█   ░█░   █▀▀ █▀▄ ██▄ █▄▀ █ █▄▄ █▄▄ █ █▄█ █░▀█

    visualizacion.mostrar_grafico_accuracy(history)
    visualizacion.mostrar_grafico_loss(history)
    
    y_pred = (model.predict(x_test) > 0.5).astype("int32")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Tuberculosis"])
    disp.plot(cmap=plt.cm.Blues)
    plt.show()




if LOAD_MODEL==True:
        
    #█▀▀ ▄▀█ █▀█ █▀▀ ▄▀█ █▀█   █▀▄▀█ █▀█ █▀▄ █▀▀ █░░ █▀█
    #█▄▄ █▀█ █▀▄ █▄█ █▀█ █▀▄   █░▀░█ █▄█ █▄▀ ██▄ █▄▄ █▄█

    model = load_model('tb_modelo_basic_560')
    print("¡Modelo cargado satisfactoriamente!")
    model.summary()

    ##########################################################################################

    def GradCAM(model, image, interpolant=0.5, plot_results=True):
        image = tf.image.rgb_to_grayscale(image)
        
        original_img = np.asarray(image, dtype=np.float32)
        img = np.expand_dims(original_img, axis=0)

        prediction = model.predict(img)
        prediction_idx = np.argmax(prediction)
  
        last_conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
                break

        if last_conv_layer is None:
            raise ValueError("No convolutional layer found in the model.")

        gradient_model = Model(inputs=model.inputs, outputs=[last_conv_layer.output, model.output])

        with tf.GradientTape() as tape:
            conv2d_out, predictions = gradient_model(img)
            loss = predictions[:, prediction_idx]

        gradients = tape.gradient(loss, conv2d_out)

        output = conv2d_out[0]
        weights = tf.reduce_mean(gradients[0], axis=(0, 1))

        activation_map = np.zeros(output.shape[0:2], dtype=np.float32)

        for idx, weight in enumerate(weights):
            activation_map += weight * output[:, :, idx]

        activation_map = cv2.resize(activation_map.numpy(), (original_img.shape[1], original_img.shape[0]))
        activation_map = np.maximum(activation_map, 0)
        activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min())
        activation_map = np.uint8(255 * activation_map)

        heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_JET)

        original_img = np.uint8((original_img - original_img.min()) / (original_img.max() - original_img.min()) * 255)
        cvt_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        if plot_results:
            plt.imshow(np.uint8(original_img * interpolant + cvt_heatmap * (1 - interpolant)))
            plt.title(f'Clase predicha: {"Tuberculosis" if prediction.max() > 0.5  else "Normal"}')
            plt.show()  
        else:
            return cvt_heatmap

    '''
    test_img = Image.open("C:/Users/vadim/Desktop/TFG/dataset_Tuberculosis/Tuberculosis/Tuberculosis-0.png")
    test_img = test_img.convert("RGB")    
    test_img = test_img.resize((512, 512))  
    test_img = np.array(test_img) / 255.0   # Normalize pixel values to [0, 1]
                
    # Preprocessed image into GradCAM
    #GradCAM(model, test_img, plot_results=True)'''
    

    # Confusion Matrix
    x_train, y_train, x_test, y_test, class_weights = datos.obtener_train_test(1)
    y_pred = (model.predict(x_test) > 0.5).astype("int32")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Tuberculosis"])
    disp.plot(cmap=plt.cm.Blues)
    plt.show()