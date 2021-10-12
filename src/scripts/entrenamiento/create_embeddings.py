import gensim 
import numpy as np
from numpy import zeros
from gensim.models.wrappers import FastText


#wordvectors2 = FastText.load_fasttext_format('data/finales/embeddings-l-model.bin') 

# Sacar el tamaño del vocabulario, usando 
vocab_size =   int(r.vocab_size)

# Crear matriz con ceros
embedding_matrix = zeros((vocab_size, 300))

# Para cada palabra dentro de mi vocabulario
for word, i in r.tokenizer.word_index.items():
  # Si el modelo predice algo para la palabra
  if word in wordvectors2: 
    # Guardar el embedding en la mtriz de ceros
    embedding_vector = wordvectors2[word] 
    embedding_matrix[i] = embedding_vector
    
    
#x = np.sum(embedding_matrix, axis = 1)
contar = 0

for w, i in r.tokenizer.word_index.items():
  if w in wordvectors2.wv.vocab:
    contar = contar + 1 
  
# Construir minimos y máximos para cada texto
