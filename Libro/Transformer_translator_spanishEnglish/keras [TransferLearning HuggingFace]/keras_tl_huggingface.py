from transformers import AutoTokenizer, TFMarianMTModel
import tensorflow as tf
from sklearn.model_selection import train_test_split

'''
!pip install transformers
!pip install sentencepiece
'''
#------------------- CREACIÓN DEL TOKENIZADOR Y DEL MODELO -------------------------

tokenizador = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
modelo = TFMarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-fr-en")

#------------------- PRUEBA DEL TOKENIZADOR IMPLEMENTADO ----------------------
tokenizador("Buenas tardes", return_tensors="tf")

'''S A L I D A  C O N S O L A
    {'input_ids': <tf.Tensor: shape=(1, 6), dtype=int32, numpy=array([[7545, 8277,
    9, 2065,  114,    0]], dtype=int32)>, 'attention_mask': <tf.Tensor: shape=(1, 6),
    dtype=int32, numpy=array([[1, 1, 1, 1, 1, 1]], dtype=int32)>}
'''
#------------------- PRUEBA DEL MODELO SIN AJUSTARSE --------------------------

entrada = tokenizador(["Espero que te haya gustado el caso practico"], return_tensors="tf").input_ids
outputs = modelo.generate(entrada)
print(tokenizador.decode(outputs[0], skip_special_tokens=True))

'''S A L I D A  C O N S O L A
    Espero que te haya gustado el caso practico
'''
#------------------- DIVISIÓN DE DATOS -----------------------------------------

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,\
                                                    test_size=0.2,\
                                                    random_state=42)
print('¡Particiones realizadas!')
print(f'Tamaño del conjunto de entrenamiento: {len(X_train)}')
print(f'Tamaño del conjunto de test: {len(X_test)}')

'''S A L I D A  C O N S O L A
    ¡Particiones realizadas!
    Tamaño del conjunto de entrenamiento: 95171
    Tamaño del conjunto de test: 23793
'''
#------------------- REIMPLEMENTAR PREPROCESADO Y GENERADOR-----------------------------------------
def preproceso_batch(X, Y, tokenizador):

   transformer_data = tokenizador(X, text_target=Y, return_tensors="tf", padding=True)

   return {"input_ids": transformer_data["input_ids"], "attention_mask": \
            transformer_data["attention_mask"], "labels": transformer_data["labels"]}

def generador_batch(X, Y, tokenizador, batch_size):
    idx = 0
    while True:
        bx = X[idx:idx+batch_size]
        by = Y[idx:idx+batch_size]

        yield preproceso_batch(bx, by, tokenizador)

        idx = (idx + batch_size) % len(X)

train_loader = generador_batch(X_train, Y_train, tokenizador=tokenizador, batch_size=16)

#------------------- COMPILAR-----------------------------------------

modelo.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=0),
               optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.98, epsilon=1e-9))

#------------------- ENTRENAR-----------------------------------------

modelo.fit(train_loader, epochs=1, steps_per_epoch=len(X_train)//16, verbose=1)

'''S A L I D A  C O N S O L A
    5948/5948 [==============================] - 1105s 168ms/step - loss: 0.6976

'''

#------------------- VER RESULTADO-----------------------------------------

entrada = tokenizador(["Espero que te haya gustado el caso practico"], return_tensors="tf").input_ids
outputs = modelo.generate(entrada)
print(tokenizador.decode(outputs[0], skip_special_tokens=True))

'''S A L I D A  C O N S O L A
    I hope you'd enjoyed the practice case.
'''
