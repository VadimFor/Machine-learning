import re
import torch
from sklearn.utils import shuffle
import torch.nn as nn
import torch.optim as optim

#█▀█ ▄▀█ █▀ █▀█   ▄█ ▀   █▀█ █▄▄ ▀█▀ █▀▀ █▄░█ █▀▀ █▀█   █▄█   ▀█▀ █▀█ ▄▀█ █▄░█ █▀ █▀▀ █▀█ █▀█ █▀▄▀█ ▄▀█ █▀█
#█▀▀ █▀█ ▄█ █▄█   ░█ ▄   █▄█ █▄█ ░█░ ██▄ █░▀█ ██▄ █▀▄   ░█░   ░█░ █▀▄ █▀█ █░▀█ ▄█ █▀░ █▄█ █▀▄ █░▀░█ █▀█ █▀▄

#█▀▄ ▄▀█ ▀█▀ █▀█ █▀
#█▄▀ █▀█ ░█░ █▄█ ▄█

'''
    wget -q http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip
    unzip -q spa-eng.zip
'''

def cargar_datos():
    with open('spa-eng/spa.txt', 'r') as f:
        lineas = f.read().splitlines()
    pares = [linea.split('\t') for linea in lineas]
    esp = [par[1] for par in pares]
    ing = [par[0] for par in pares]
    return esp, ing

src, tgt = cargar_datos()
print(f'Número de pares de oraciones: {len(src)}')
print(f'Posible entrada: {src[50]}')
print(f'Posible salida: {tgt[50]}')


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

src_w2i, src_i2w = crear_vocab(src)
tgt_w2i, tgt_i2w = crear_vocab(tgt)
print(f'Tamaño del vocabulario de español: {len(src_w2i)}')
print(f'Tamaño del vocabulario de inglés: {len(tgt_w2i)}')

'''ＳＡＬＩＤＡ ＣＯＮＳＯＬＡ:
    Tamaño del vocabulario de español: 28993
    Tamaño del vocabulario de inglés: 14779
'''

def codificar_secuencias(secs, w2i):
    secs_cod = []
    for s in secs:
        s_cod = [w2i[w] for w in re.findall(r'\w+|[^\w\s]', s)]
        s_cod = [w2i['SOS']] + s_cod + [w2i['EOS']]
        secs_cod.append(s_cod)
    return secs_cod

src_cod = codificar_secuencias(src, src_w2i)
tgt_cod = codificar_secuencias(tgt, tgt_w2i)


from sklearn.model_selection import train_test_split

src_train, src_test, tgt_train, tgt_test = train_test_split(src_cod, tgt_cod,\
                                                    test_size=0.2,\
                                                    random_state=42)
print('¡Particiones realizadas!')
print(f'Tamaño del conjunto de entrenamiento: {len(src_train)}')
print(f'Tamaño del conjunto de test: {len(src_test)}')

'''ＳＡＬＩＤＡ ＣＯＮＳＯＬＡ:
    ¡Particiones realizadas!
    Tamaño del conjunto de entrenamiento: 95171
    Tamaño del conjunto de test: 23793
'''

####################PREPROCESADO DE LOS DATOS DE ENTRENAMIENTO
def preproceso_batch(X, Y):
    max_len_X = max([len(x) for x in X])
    max_len_Y = max([len(y) for y in Y])

    encoder_input = torch.zeros(len(X), max_len_X)
    decoder_input = torch.zeros(len(Y), max_len_Y)
    salida = torch.zeros(len(Y), max_len_Y)

    for i, s in enumerate(X):
        # Sec. completa con relleno para el encoder (frase a traducir)
        encoder_input[i, :len(s)] = torch.tensor(s)

    for i, s in enumerate(Y):
        # Sec. sin el "EOS" con relleno para el decoder (traducción)
        decoder_input[i, :len(s)-1] = torch.tensor(s[:-1])
        # Sec. sin el "SOS" con relleno para la salida (traducción)
        salida[i, :len(s)-1] = torch.tensor(s[1:])

    return encoder_input.long(), decoder_input.long(), salida.long()

def generador_batch(X, Y, batch_size):
    idx = 0
    while True:
        bx = X[idx:idx+batch_size]
        by = Y[idx:idx+batch_size]

        yield preproceso_batch(bx, by)

        idx = (idx + batch_size) % len(X)
        if idx == 0:
            X, Y = shuffle(X, Y, random_state=42)

batch_size = 128
train_loader = generador_batch(src_train, tgt_train, batch_size=batch_size)
be, bd, bs = next(train_loader)
print(f'Entrada al encoder: {[src_i2w[w.item()]for w in be[0]]}')
print(f'Entrada al decoder: {[tgt_i2w[w.item()]for w in bd[0]]}')
print(f'Salida del decoder: {[tgt_i2w[w.item()]for w in bs[0]]}')


'''ＳＡＬＩＤＡ ＣＯＮＳＯＬＡ:
    Entrada al encoder: ['SOS', 'No', 'tengo', 'otra', 'opción', 'en', 'absoluto', '.', 'EOS', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD']
    Entrada al decoder: ['SOS', 'I', 'have', 'no', 'choice', 'at', 'all', '.', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD']
    Salida del decoder: ['I', 'have', 'no', 'choice', 'at', 'all', '.', 'EOS', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD']
'''

#█▀█ ▄▀█ █▀ █▀█   ▀█ ▀   █▀▄ █▀▀ █▀▀ █ █▄░█ █ █▀█   ▄▀█ █▀█ █▀█ █░█ █ ▀█▀ █▀▀ █▀▀ ▀█▀ █░█ █▀█ ▄▀█
#█▀▀ █▀█ ▄█ █▄█   █▄ ▄   █▄▀ ██▄ █▀░ █ █░▀█ █ █▀▄   █▀█ █▀▄ ▀▀█ █▄█ █ ░█░ ██▄ █▄▄ ░█░ █▄█ █▀▄ █▀█

#DEFINICION DE LA CAPA DE CODIFICACION DE POSICION
class PositionalEncoding(nn.Module):
    def __init__(self, max_len, emb_dim, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pos = torch.arange(max_len).unsqueeze(1)
        # 2 * torch.arange(emb_dim // 2)  == torch.arange(0, emb_dim, 2)
        den = torch.pow(10000, torch.arange(0, emb_dim, 2) / emb_dim)
        pe = torch.zeros(1, max_len, emb_dim)
        pe[0, :, 0::2] = torch.sin(pos / den)
        pe[0, :, 1::2] = torch.cos(pos / den)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x.shape = [batch_size, sec_len, emb_dim]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

#DEFINICIÓN DEL MODELO TRANSFORMER A USAR
class Transformer(nn.Module):
    def __init__(self,
                 max_len,
                 emb_dim,
                 num_encoder_layers,
                 num_decoder_layers,
                 nhead,
                 src_vocab_size,
                 tgt_vocab_size,
                 dim_feedforward,
                 dropout=0.1):
        super(Transformer, self).__init__()

        # Capas de embedding + codificación de posición
        self.src_embedding = nn.Embedding(src_vocab_size, emb_dim)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, emb_dim)
        self.pe = PositionalEncoding(max_len, emb_dim, dropout)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=emb_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # Capa de salida
        self.salida = nn.Linear(emb_dim, tgt_vocab_size)

    def forward(self, src, tgt):
        # Máscara de padding
        src_mask, tgt_mask,\
            src_pad_mask, tgt_pad_mask = self.crear_mascara(src, tgt)
        # Embedding + codificación de posición
        src_emb = self.pe(self.src_embedding(src))
        tgt_emb = self.pe(self.tgt_embedding(tgt))
        # Transformer
        tgt_pred = self.transformer(src_emb, tgt_emb,
                                    src_mask=src_mask,
                                    tgt_mask=tgt_mask,
                                    src_key_padding_mask=src_pad_mask,
                                    tgt_key_padding_mask=tgt_pad_mask,
                                    memory_key_padding_mask=src_pad_mask)
        # Salida (clasificación)
        tgt_pred = self.salida(tgt_pred)
        return tgt_pred

    def codificar(self, src, src_mask):
        # Embedding + codificación de posición
        src_emb = self.pe(self.src_embedding(src))
        # Transformer encoder
        return self.transformer.encoder(src_emb, src_mask)

    def decodificar(self, tgt, memory, tgt_mask):
        # Embedding + codificación de posición
        tgt_emb = self.pe(self.tgt_embedding(tgt))
        # Transformer decoder
        tgt_pred = self.transformer.decoder(tgt_emb, memory, tgt_mask)
        # Salida (clasificación)
        tgt_pred = self.salida(tgt_pred)
        return tgt_pred

    def crear_mascara(self, src, tgt):
        # src/tgt.shape = [batch_size, src/tgt_sec_len, emb_dim]
        src_sec_len = src.shape[1]
        tgt_sec_len = tgt.shape[1]

        # Máscara de ceros (dejamos ver todo)
        src_mask = torch.zeros((src_sec_len, src_sec_len),
                               device=src.device)
        # Máscara triangular superior para el target
        tgt_mask = self.transformer.generate_square_subsequent_mask(
            tgt_sec_len, tgt.device)

        # 0 == "PAD"
        src_pad_mask = (src == 0)
        tgt_pad_mask = (tgt == 0)

        return src_mask, tgt_mask, src_pad_mask, tgt_pad_mask


# Comprobamos si tenemos GPU
dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Calculamos la máxima longitud de las frases
max_len = max([len(x) for x in src + tgt])

# Hiperparámetros del artículo del Transformer
modelo = Transformer(
    max_len=max_len,
    emb_dim=512,
    num_encoder_layers=6,
    num_decoder_layers=6,
    nhead=8,
    src_vocab_size=len(src_w2i),
    tgt_vocab_size=len(tgt_w2i),
    dim_feedforward=2048,
    dropout=0.1
).to(dispositivo)


#█▀█ ▄▀█ █▀ █▀█  ▀█ ▀   █▀▀ █▀█ █▀▄▀█ █▀█ █ █░░ ▄▀█ █▀█   █▄█   █▀▀ █▄░█ ▀█▀ █▀█ █▀▀ █▄░█ ▄▀█ █▀█
#█▀▀ █▀█ ▄█ █▄█  ▄█ ▄   █▄▄ █▄█ █░▀░█ █▀▀ █ █▄▄ █▀█ █▀▄   ░█░   ██▄ █░▀█ ░█░ █▀▄ ██▄ █░▀█ █▀█ █▀▄

#█▀▄▀█ █▀█ █▀▄ █▀▀ █░░ █▀█
#█░▀░█ █▄█ █▄▀ ██▄ █▄▄ █▄█


# Pérdida y optimizador
func_perdida = nn.CrossEntropyLoss(ignore_index=0)
optimizador = optim.Adam(modelo.parameters(),
                         lr=0.0001,
                         betas=(0.9, 0.98),
                         eps=1e-9)

modelo.train()
epocas = 20
for epoca in range(epocas):
    epoca_perdidas = []
    for _ in range(len(src_train) // batch_size):
        # Obtenemos un lote de datos
        be, bd, bs = next(train_loader)
        # Enviamos los datos al dispositivo (GPU o CPU)
        be = be.to(dispositivo)
        bd = bd.to(dispositivo)
        bs = bs.to(dispositivo)
        # Calculamos la pérdida y
        # actualizamos los parámetros
        optimizador.zero_grad()
        bs_pred = modelo(be, bd)
        perdida = func_perdida(bs_pred.permute(0, 2, 1), bs)
        perdida.backward()
        optimizador.step()
        # Guardamos la pérdida
        epoca_perdidas.append(perdida.detach().cpu().item())
    # Pérdida promedio de la época
    perdida_epoca = sum(epoca_perdidas) / len(epoca_perdidas)
    print(f'Época {epoca+1}/{epocas}, pérdida={perdida_epoca:.4f}')
print('Fin del entrenamiento.')

'''ＳＡＬＩＤＡ ＣＯＮＳＯＬＡ:
    Época 1/20,          pérdida=4.0026,
    Época 2/20,          pérdida=2.8578,
    Época 3/20,          pérdida=2.3689,
    Época 4/20,          pérdida=2.0082,
    Época 5/20,          pérdida=1.7391,
    Época 6/20,          pérdida=1.5368,
    Época 7/20,          pérdida=1.3748,
    Época 8/20,          pérdida=1.2430,
    Época 9/20,          pérdida=1.1319,
    Época 10/20,          pérdida=1.0354,
    Época 11/20,          pérdida=0.9523,
    Época 12/20,          pérdida=0.8774,
    Época 13/20,          pérdida=0.8111,
    Época 14/20,          pérdida=0.7498,
    Época 15/20,          pérdida=0.6945,
    Época 16/20,          pérdida=0.6454,
    Época 17/20,          pérdida=0.5984,
    Época 18/20,          pérdida=0.5581,
    Época 19/20,          pérdida=0.5195,
    Época 20/20,          pérdida=0.4840,
    Fin del entrenamiento.
'''


#█▀█ ▄▀█ █▀ █▀█   █░█ ▀   █▀▄ █▀▀ █▀▀ █▀█ █▀▄ █ █▀▀ █ █▀▀ ▄▀█ █▀█
#█▀▀ █▀█ ▄█ █▄█   ▀▀█ ▄   █▄▀ ██▄ █▄▄ █▄█ █▄▀ █ █▀░ █ █▄▄ █▀█ █▀▄

def decodificacion_voraz(modelo,
                         src, src_mask,
                         max_len,
                         tgt_w2i, tgt_i2w,
                         dispositivo):
    # Codificación
    src_cod = modelo.codificar(src, src_mask)

    # Decodificación
    tgt_token = torch.tensor([tgt_w2i['SOS']]).unsqueeze(0).long()
    tgt_token = tgt_token.to(dispositivo)

    tgt_pred_decod = []
    for i in range(max_len):
        # Predicción del modelo
        tgt_mask = modelo.transformer.generate_square_subsequent_mask(
            tgt_token.size(1), dispositivo)
        tgt_pred = modelo.decodificar(tgt_token, src_cod, tgt_mask)
        tgt_pred = tgt_pred[0, -1, :] # Último token

        # Nos quedamos con el token más probable
        tgt_pred = tgt_pred.argmax(dim=-1).item()
        tgt_pred_decod.append(tgt_i2w[tgt_pred])

        # Preparamos la nueva entrada del decoder
        tgt_token = torch.cat((tgt_token,
                               torch.full((1, 1),
                                          tgt_pred,
                                          device=dispositivo)),
                               dim=1)

        # Comprobamos si se ha predicho el token de
        # fin de secuencia
        if tgt_pred_decod[-1] == 'EOS':
            break

    return tgt_pred_decod

def traducir(modelo,
             src_frase, src_w2i,
             tgt_w2i, tgt_i2w,
             dispositivo):
    # Codificamos la secuencia de entrada
    src_cod = codificar_secuencias([src_frase], src_w2i)
    src_cod = torch.tensor(src_cod).long().to(dispositivo) # [1, sec_len]
    # Máscara de ceros para el soruce (dejamos ver todo)
    src_mask = torch.zeros((src_cod.size(1), src_cod.size(1)),
                           device=dispositivo)

    # Permitimos hasta 5 tokens más en la traducción
    max_len = src_cod.size(1) + 5

    # Iniciamos la traducción
    modelo.eval()
    with torch.no_grad():
        tgt_pred_decod = decodificacion_voraz(
            modelo,
            src_cod,
            src_mask,
            max_len,
            tgt_w2i,
            tgt_i2w,
            dispositivo)
    # Quitamos los tokens de inicio y fin de secuencia
    tgt_pred_decod = [t for t in tgt_pred_decod if t not in ['SOS', 'EOS']]
    return ' '.join(tgt_pred_decod)


src_frase = 'Espero que te haya gustado el caso de estudio'
tgt_frase = traducir(
    modelo,
    src_frase,
    src_w2i,
    tgt_w2i, tgt_i2w,
    dispositivo
)
print(f'Original: {src_frase}\nTraducción: {tgt_frase}')

'''ＳＡＬＩＤＡ ＣＯＮＳＯＬＡ:
   Original: Espero que te haya gustado el caso de estudio
    Traducción: I hope you ' ll like the study of the study .
'''
