import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

######################################################################################################################################################
#▀▄▀ ▀█▀ █▀▀ █▀ ▀█▀   ▀▄▀ ▀█▀ █▀█ ▄▀█ █ █▄░█
#█░█ ░█░ ██▄ ▄█ ░█░   █░█ ░█░ █▀▄ █▀█ █ █░▀█

def mostrarDimensiones(x_1,y_1,x_test_normal, x_2,y_2,x_test_tuberculosis):
    print(f"Forma de x_train_normal: {x_1.shape} | Forma de y_train_normal: {y_1.shape}")
    print(f"Forma de x_train_tuberculosis: {x_2.shape} | Forma de y_train_tuberculosis: {y_2.shape}")
    print(f"Forma de test_Normal: {x_test_normal.shape} | Forma de test_tuberculosis: {x_test_tuberculosis.shape}")


#Visualizar una imagen con su etiqueta
def mostrar_imagen_etiqueta(x_train, y_train):
    plt.imshow(np.squeeze(x_train[0]), cmap='gray')
    plt.title(f'Etiqueta: {y_train[0]}')
    plt.axis('off')
    plt.show()
    
# Visualizar cantidad imágenes de cada clase
def mostrar_proporciones_clases(x_train_normal, x_train_tuberculosis):
    plt.figure(figsize=(8, 6))
    plt.bar(["Normal", "Tuberculosis"], [len(x_train_normal), len(x_train_tuberculosis)], color=['Green', 'red'])
    plt.title("Distribución de clase")
    plt.xlabel("Clase")
    plt.ylabel("Cantidad de imágenes")
    plt.show()
    
def mostrar_f1(model,x_test,y_test):
    y_pred_probs = model.predict(x_test) 
    y_pred_labels = (y_pred_probs > 0.5).astype(int) 
    print(classification_report(y_test, y_pred_labels, digits=4, zero_division=0))
  
######################################################################################################################################################      
#█░█ █ █▀ ▀█▀ █▀█ █▀█ █▄█
#█▀█ █ ▄█ ░█░ █▄█ █▀▄ ░█░

def mostrar_grafico_accuracy(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()
    
def mostrar_grafico_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim([0, 1])
    plt.legend(loc='upper right')
    plt.show()