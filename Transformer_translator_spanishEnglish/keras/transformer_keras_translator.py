import keras_nlp
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import numpy as np
import tensorflow as tf



#█▀█ ▄▀█ █▀ █▀█   ▄█ ▀   █▀█ █▄▄ ▀█▀ █▀▀ █▄░█ █▀▀ █▀█   █▄█   ▀█▀ █▀█ ▄▀█ █▄░█ █▀ █▀▀ █▀█ █▀█ █▀▄▀█ ▄▀█ █▀█
#█▀▀ █▀█ ▄█ █▄█   ░█ ▄   █▄█ █▄█ ░█░ ██▄ █░▀█ ██▄ █▀▄   ░█░   ░█░ █▀▄ █▀█ █░▀█ ▄█ █▀░ █▄█ █▀▄ █░▀░█ █▀█ █▀▄

#█▀▄ ▄▀█ ▀█▀ █▀█ █▀
#█▄▀ █▀█ ░█░ █▄█ ▄█

'''
    !wget -q http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip
    !unzip -q spa-eng.zip
    !pip install -q git+https://github.com/keras-team/keras-nlp.git --upgrade
'''
def cargar_datos():
    with open('spa-eng/spa.txt', 'r') as f:
        lineas = f.read().splitlines()
    pares = [linea.split('\t') for linea in lineas]
    esp = [par[1] for par in pares]
    ing = [par[0] for par in pares]
    return esp, ing

X, Y = cargar_datos()
print(f'Número de pares de oraciones: {len(X)}')
print(f'Posible entrada: {X[50]}')
print(f'Posible salida: {Y[50]}')

'''ＳＡＬＩＤＡ ＣＯＮＳＯＬＡ:
    Número de pares de oraciones: 118964
    Posible entrada: Estoy levantado.
    Posible salida: I'm up.
'''

def crear_vocab(frases):
   # Obtenemos el vocabulario
   vocab = set()
   for f in frases:
       # Expresión regular para separar palabras
       # manteniendo signos de puntuación
       vocab.update(re.findall(r'\w+|[^\w\s]', f))

   # Creamos los diccionarios
   w2i = {w: i+4 for i, w in enumerate(vocab)}
   w2i['PAD'] = 0
   w2i['SOS'] = 1
   w2i['EOS'] = 2
   w2i['UNK'] = 3
   i2w = {i: w for w, i in w2i.items()}

   return w2i, i2w

X_w2i, X_i2w = crear_vocab(X)
Y_w2i, Y_i2w = crear_vocab(Y)
print(f'Tamaño del vocabulario de español: {len(X_w2i)}')
print(f'Tamaño del vocabulario de inglés: {len(Y_w2i)}')

'''ＳＡＬＩＤＡ ＣＯＮＳＯＬＡ:
    Tamaño del vocabulario de español: 28993
    Tamaño del vocabulario de inglés: 14779
'''

def codificar_secuencia(secs, w2i):
    secs_cod = []
    for s in secs:
        s_cod = [w2i[w] for w in re.findall(r'\w+|[^\w\s]', s)]
        s_cod = [w2i['SOS']] + s_cod + [w2i['EOS']]
        secs_cod.append(s_cod)
    return secs_cod

X_cod = codificar_secuencia(X, X_w2i)
Y_cod = codificar_secuencia(Y, Y_w2i)

X_train, X_test, Y_train, Y_test = train_test_split(X_cod, Y_cod,\
                                                    test_size=0.2,\
                                                    random_state=42)
print('¡Particiones realizadas!')
print(f'Tamaño del conjunto de entrenamiento: {len(X_train)}')
print(f'Tamaño del conjunto de test: {len(X_test)}')

'''ＳＡＬＩＤＡ ＣＯＮＳＯＬＡ:
    ¡Particiones realizadas!
    Tamaño del conjunto de entrenamiento: 95171
    Tamaño del conjunto de test: 23793
'''

def preproceso_batch(X, Y):
   max_len_X = max([len(x) for x in X])
   max_len_Y = max([len(y) for y in Y])

   encoder_input = np.zeros((len(X), max_len_X))
   decoder_input = np.zeros((len(Y), max_len_Y))
   salida = np.zeros((len(Y), max_len_Y))

   for i, s in enumerate(X):
       # Sec. completa con relleno para el encoder (frase a traducir)
       encoder_input[i, :len(s)] = np.array(s)

   for i, s in enumerate(Y):
       # Sec. sin el "EOS" con relleno para el decoder (traducción)
       decoder_input[i, :len(s)-1] = np.array(s[:-1])
       # Sec. sin el "SOS" con relleno para la salida (traducción)
       salida[i, :len(s)-1] = np.array(s[1:])

   src_pad_mask = (encoder_input == 0)
   tgt_pad_mask = (decoder_input == 0)

   encoder_input = encoder_input.astype(np.int64)
   decoder_input = decoder_input.astype(np.int64)
   salida = salida.astype(np.int64)

   return [encoder_input, decoder_input, src_pad_mask, tgt_pad_mask], salida

def generador_batch(X, Y, batch_size):
    idx = 0
    while True:
        bx = X[idx:idx+batch_size]
        by = Y[idx:idx+batch_size]

        yield preproceso_batch(bx, by)

        idx = (idx + batch_size) % len(X)

batch_size = 128
train_loader = generador_batch(X_train, Y_train, batch_size=batch_size)
[be, bd, sp, tp], bs = next(train_loader)
print(f'Entrada al encoder: {[X_i2w[w.item()]for w in be[0]]}')
print(f'Entrada al decoder: {[Y_i2w[w.item()]for w in bd[0]]}')
print(f'Salida del decoder: {[Y_i2w[w.item()]for w in bs[0]]}')
print(f'Mascara del encoder: {sp[0]}')
print(f'Mascara del decoder: {tp[0]}')

'''ＳＡＬＩＤＡ ＣＯＮＳＯＬＡ:
    Entrada al encoder: ['SOS', 'No', 'tengo', 'otra', 'opción', 'en', 'absoluto', '.', 'EOS', 'PAD', 
    'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD']
    Entrada al decoder: ['SOS', 'I', 'have', 'no', 'choice', 'at', 'all', '.', 'PAD', 'PAD', 'PAD', 
    'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD']
    Salida del decoder: ['I', 'have', 'no', 'choice', 'at', 'all', '.', 'EOS', 'PAD', 'PAD', 'PAD', 
    'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD']
    Mascara del encoder: [False False False False False False False False False  True  True  True
    True  True  True  True  True  True]
    Mascara del decoder: [False False False False False False False False  True  True  True  True
    True  True  True  True  True  True  True  True  True]
'''

#█▀█ ▄▀█ █▀ █▀█   ▀█ ▀   █▀▄ █▀▀ █▀▀ █ █▄░█ █ █▀█   ▄▀█ █▀█ █▀█ █░█ █ ▀█▀ █▀▀ █▀▀ ▀█▀ █░█ █▀█ ▄▀█
#█▀▀ █▀█ ▄█ █▄█   █▄ ▄   █▄▀ ██▄ █▀░ █ █░▀█ █ █▀▄   █▀█ █▀▄ ▀▀█ █▄█ █ ░█░ ██▄ █▄▄ ░█░ █▄█ █▀▄ █▀█


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_len, emb_dim, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)

        pos = np.arange(max_len).reshape(-1, 1)
        den = np.power(10000, np.arange(0, emb_dim, 2) / emb_dim)
        pe = np.zeros((1, max_len, emb_dim))
        pe[0, :, 0::2] = np.sin(pos / den)
        pe[0, :, 1::2] = np.cos(pos / den)
        self.pe = tf.constant(pe, dtype=tf.float32)

    def call(self, x):
        # x.shape = [batch_size, sec_len, emb_dim]
        x = x + self.pe[:, :tf.shape(x)[1], :]
        return self.dropout(x)
def crear_transformer(max_long,
                      emb_dim,
                      num_enc_capas,
                      num_dec_capas,
                      ncabezas,
                      src_vocab_tam,
                      tgt_vocab_tam,
                      dim_mlp,
                      dropout=0.1):

  # Definicion del encoder
  enc_entradas = tf.keras.Input(shape=(None,),
                                dtype="int64",
                                name="enc_entradas")
  mask_entradas_encoder = tf.keras.Input(shape=(None,),
                                         dtype="int64",
                                         name="mask_entradas_encoder")

  enc_salidas = tf.keras.layers.Embedding(src_vocab_tam, emb_dim)(enc_entradas)
  enc_salidas = PositionalEncoding(max_long, emb_dim, 0.1)(enc_salidas)

  for i in range(num_enc_capas):
       enc_salidas = keras_nlp.layers.TransformerEncoder(
           intermediate_dim=dim_mlp,
           num_heads=ncabezas,
           dropout=dropout,
           activation="relu",
           name=None)(enc_salidas, padding_mask=mask_entradas_encoder)

  # Definicion del decoder
  dec_entradas = tf.keras.Input(shape=(None,), dtype="int64",
                      name="dec_entradas")
  enc_seq_entradas = tf.keras.Input(shape=(None, emb_dim),
                          name="dec_state_entradas")

  mask_entradas_decoder = tf.keras.Input(shape=(None,),
                                         dtype="int64",
                                         name="mask_entradas_decoder")

  dec_salidas = tf.keras.layers.Embedding(tgt_vocab_tam, emb_dim)(dec_entradas)
  dec_salidas = PositionalEncoding(max_long, emb_dim, 0.1)(dec_salidas)

  capas_decoder = []
  for _ in range(num_dec_capas):
    capas_decoder.append(keras_nlp.layers.TransformerDecoder(
           intermediate_dim=dim_mlp,
           num_heads=ncabezas,
           dropout=dropout,
           activation="relu",
           name=None))

  trf_salida = dec_salidas

  for capa in capas_decoder:
       trf_salida = capa(decoder_sequence=trf_salida,
                         encoder_sequence=enc_salidas,
                         decoder_padding_mask=mask_entradas_decoder,
                         use_causal_mask=True)

  for capa in capas_decoder:
       dec_salidas = capa(decoder_sequence=dec_salidas,
                          encoder_sequence=enc_seq_entradas,
                          decoder_padding_mask=mask_entradas_decoder,
                          use_causal_mask=True)

  capa_salida = tf.keras.layers.Dense(tgt_vocab_tam,
                    activation="linear")

  salida_transformer = capa_salida(trf_salida)
  salida_decoder = capa_salida(dec_salidas)

  # Definicion del Transformer
  encoder = tf.keras.Model([enc_entradas, mask_entradas_encoder],
                          enc_salidas,
                          name="encoder",
  )

  decoder = tf.keras.Model([dec_entradas, enc_seq_entradas, mask_entradas_decoder],
                          salida_decoder,
                          name="decoder",
  )

  transformer = tf.keras.Model([enc_entradas, dec_entradas, mask_entradas_encoder, mask_entradas_decoder],
                          salida_transformer,
                          name="transformer",
  )

  transformer.summary()
  perdida = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=0)
  optimizador = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
  transformer.compile(loss=perdida, optimizer=optimizador)

  return transformer, encoder, decoder
       
'''ＳＡＬＩＤＡ ＣＯＮＳＯＬＡ:
Model: "transformer"
__________________________________________________________________________________________________
 Layer (type)                Output Shape                 Param #   Connected to                  
==================================================================================================
 enc_entradas (InputLayer)   [(None, None)]               0         []                            
                                                                                                  
 embedding (Embedding)       (None, None, 512)            1484441   ['enc_entradas[0][0]']        
                                                          6                                       
                                                                                                  
 positional_encoding (Posit  (None, None, 512)            0         ['embedding[0][0]']           
 ionalEncoding)                                                                                   
                                                                                                  
 mask_entradas_encoder (Inp  [(None, None)]               0         []                            
 utLayer)                                                                                         
                                                                                                  
 transformer_encoder (Trans  (None, None, 512)            3152384   ['positional_encoding[0][0]', 
 formerEncoder)                                                      'mask_entradas_encoder[0][0]'
                                                                    ]                             
                                                                                                  
 transformer_encoder_1 (Tra  (None, None, 512)            3152384   ['transformer_encoder[0][0]', 
 nsformerEncoder)                                                    'mask_entradas_encoder[0][0]'
                                                                    ]                             
                                                                                                  
 transformer_encoder_2 (Tra  (None, None, 512)            3152384   ['transformer_encoder_1[0][0]'
 nsformerEncoder)                                                   , 'mask_entradas_encoder[0][0]
...
Total params: 74131387 (282.79 MB)
Trainable params: 74131387 (282.79 MB)
Non-trainable params: 0 (0.00 Byte)
'''
##############INSTANCIACIÓN DEL TRANSFORMER
max_long = max([len(x) for x in X + Y])
# Instancia del modelo Transformer
transformer, encoder, decoder = crear_transformer(
   max_long=max_long,
   emb_dim=512,
   num_enc_capas=6,
   num_dec_capas=6,
   ncabezas=8,
   src_vocab_tam=len(X_w2i),
   tgt_vocab_tam=len(Y_w2i),
   dim_mlp=2048,
   dropout=0.1
)

#█▀█ ▄▀█ █▀ █▀█  ▀█ ▀   █▀▀ █▀█ █▀▄▀█ █▀█ █ █░░ ▄▀█ █▀█   █▄█   █▀▀ █▄░█ ▀█▀ █▀█ █▀▀ █▄░█ ▄▀█ █▀█
#█▀▀ █▀█ ▄█ █▄█  ▄█ ▄   █▄▄ █▄█ █░▀░█ █▀▀ █ █▄▄ █▀█ █▀▄   ░█░   ██▄ █░▀█ ░█░ █▀▄ ██▄ █░▀█ █▀█ █▀▄

#█▀▄▀█ █▀█ █▀▄ █▀▀ █░░ █▀█
#█░▀░█ █▄█ █▄▀ ██▄ █▄▄ █▄█

epocas = 20
train_loader = generador_batch(X_train, Y_train, batch_size=128)
transformer.fit(train_loader, epochs=epocas, steps_per_epoch=len(X_train)//128, verbose=1)

'''ＳＡＬＩＤＡ ＣＯＮＳＯＬＡ:
    Epoch 1/20
    743/743 [==============================] - 350s 384ms/step - loss: 5.3182
    Epoch 2/20
    743/743 [==============================] - 265s 357ms/step - loss: 5.3015
    Epoch 3/20
    743/743 [==============================] - 263s 354ms/step - loss: 5.1645
    Epoch 4/20
    743/743 [==============================] - 264s 355ms/step - loss: 4.8761
    Epoch 5/20
    743/743 [==============================] - 263s 354ms/step - loss: 4.7273
    Epoch 6/20
    743/743 [==============================] - 263s 354ms/step - loss: 4.8893
    Epoch 7/20
    743/743 [==============================] - 264s 356ms/step - loss: 4.6304
    Epoch 8/20
    743/743 [==============================] - 264s 355ms/step - loss: 4.5480
    Epoch 9/20
    743/743 [==============================] - 264s 355ms/step - loss: 4.4777
    Epoch 10/20
    743/743 [==============================] - 263s 354ms/step - loss: 4.3213
    Epoch 11/20
    743/743 [==============================] - 263s 354ms/step - loss: 4.2240
    Epoch 12/20
    743/743 [==============================] - 262s 353ms/step - loss: 4.2508
    Epoch 13/20
    ...
    Epoch 19/20
    743/743 [==============================] - 262s 353ms/step - loss: 4.0705
    Epoch 20/20
    743/743 [==============================] - 261s 352ms/step - loss: 4.0213
'''


#█▀█ ▄▀█ █▀ █▀█   █░█ ▀   █▀▄ █▀▀ █▀▀ █▀█ █▀▄ █ █▀▀ █ █▀▀ ▄▀█ █▀█
#█▀▀ █▀█ ▄█ █▄█   ▀▀█ ▄   █▄▀ ██▄ █▄▄ █▄█ █▄▀ █ █▀░ █ █▄▄ █▀█ █▀▄


def decodificacion_voraz(codificador, decodificador, src, src_mask, max_len, tgt_w2i, tgt_i2w):
   # Codificación
   print(src.shape)
   print(src_mask.shape)

   src_cod = codificador.predict([src, src_mask], verbose=0)

   # Decodificación
   tgt_token = tf.constant([[tgt_w2i['SOS']]], dtype=tf.int64)

   tgt_pred_decod = []
   for i in range(max_len):
       # Predicción del modelo
       tgt_pred = decodificador.predict([tgt_token, src_cod, (tgt_token == 0)], verbose=0)
       tgt_pred = tgt_pred[:, -1, :]  # Último token

       # Nos quedamos con el token más probable
       tgt_pred = tf.argmax(tgt_pred, axis=-1).numpy()[0]
       tgt_pred_decod.append(tgt_i2w[tgt_pred])

       print(f'token predicho: {tgt_pred}')
       print(f'secuencia: {tgt_pred_decod}')

       # Preparamos la nueva entrada del decoder
       tgt_token = np.hstack((tgt_token, np.array([[tgt_pred]])))

       # Comprobamos si se ha predicho el token de fin de secuencia
       if tgt_pred_decod[-1] == 'EOS':
           break

   return tgt_pred_decod


def traducir(codificador, decodificador, src_frase, src_w2i, tgt_w2i, tgt_i2w):
   # Codificamos la secuencia de entrada
   src_cod = codificar_secuencia([src_frase], src_w2i)
   src_cod = tf.convert_to_tensor(src_cod, dtype=tf.int64)
   # src_cod = tf.expand_dims(src_cod, axis=0)  # Agregamos dimensión de batch [1, sec_len]

   # Máscara de ceros para el source (dejamos ver todo)
   src_mask = tf.zeros((1, src_cod.shape[1]))

   # Permitimos hasta 5 tokens más en la traducción
   max_len = src_cod.shape[1] + 5

   # Iniciamos la traducción
   tgt_pred_decod = decodificacion_voraz(codificador, decodificador, src_cod, src_mask, max_len, tgt_w2i, tgt_i2w)

   # Quitamos los tokens de inicio y fin de secuencia
   tgt_pred_decod = [t for t in tgt_pred_decod if t not in ['SOS', 'EOS']]
   return ' '.join(tgt_pred_decod)


src_frase = 'Espero que te haya gustado el caso de estudio'
tgt_frase = traducir(
   encoder, decoder,
   src_frase,
   X_w2i,
   Y_w2i, Y_i2w,
)
'''ＳＡＬＩＤＡ ＣＯＮＳＯＬＡ:
(1, 11)
(1, 11)
token predicho: 4574
secuencia: ['He']
token predicho: 11534
secuencia: ['He', 'is']
token predicho: 2
secuencia: ['He', 'is', 'EOS']
'''