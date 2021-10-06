
library(keras)
library(reticulate)
library(tidyverse)
library(caret)
library(feather)
library(tensorflow)

# Cargar funciones
source("scripts/entrenamiento/helpers_training.R", encoding = "utf-8")

# Cargar datos
path = "data/finales/train_ciuo.feather"
df = read_feather(path)

# Aquí están los inputs para la red
network_inputs <- pre_process_ciuo(df, text_variable =  "glosa_dep", label =  "gran_grupo", type = "sequences")

# Parámetros de la red
vocab_size = length(network_inputs$tokenizer$word_index) + 1  
embedding_dim = 300
maxlen = 40
clases = length(unique(network_inputs$y_train)) 

# Crear los embeddings a partir del modelo gensim. Tarda en correr, debido a que el modelo es muy pesado
tokenizer <- network_inputs$tokenizer

source_python("scripts/entrenamiento/red/create_embeddings.py")

# Matriz de embeddings creada con código python
embedding_matrix <-  py$embedding_matrix
dim(embedding_matrix)

##########################
# Arquitectura de la red #
##########################

input = layer_input(shape = list(maxlen), name = "input")

model = input %>%
  layer_embedding(input_dim = vocab_size, output_dim = 300, input_length = maxlen,
                  weights = list(embedding_matrix), trainable = FALSE) %>%
  layer_spatial_dropout_1d(rate = 0.1 ) %>%
  #bidirectional(
  layer_gru(units = 150, return_sequences = TRUE)
#)
max_pool = model %>% layer_global_max_pooling_1d()
ave_pool = model %>% layer_global_average_pooling_1d()

output = layer_concatenate(list(ave_pool, max_pool)) %>%
  layer_dense(300, activation='relu') %>%
  layer_dropout(0.5) %>%
  layer_dense(units = clases, activation = "softmax")

model = keras_model(input, output)


# Compilar modelo
model %>% 
  compile(
    loss = "sparse_categorical_crossentropy", 
    optimizer = "adam",
    metrics = 'accuracy')

summary(model)

# Entrenar
tf$random$set_seed(104) 
history <- model %>% fit(
  network_inputs$x_train, network_inputs$y_train,
  epochs = 25,
  validation_data = list(network_inputs$x_test, network_inputs$y_test),
  batch_size = 256,
  verbose = TRUE
)

# Save the model
save_model_hdf5(model, "data/finales/modelos/modelo_red_emb_gru_1d_ciuo")

# Save keys to recover original labels
write_feather(network_inputs$keys, "data/finales/modelos/keys_1d_ciuo.feather")


###########
# TESTING #
###########

# Recreate the exact same model purely from the file
model <- load_model_hdf5("data/finales/modelos/modelo_red_emb_gru_1d")
tokenizer <- load_text_tokenizer("data/finales/modelos/tokenizer_1d")

path = "data/finales/test.feather"
df2 = read_feather(path)

df2 <- df2 %>% 
  filter(!is.na(seccion)) 


# # Generar la tokenización
X_test2 = texts_to_sequences(tokenizer, df2$glosa_dep)

# # Hacer padding para que llenar con ceros e igualar el largo de los vectores
maxlen = 40
X_test2 = pad_sequences(X_test2, padding='post', maxlen=maxlen)
# 

# Probability distribution for all classes
class_pred <- model %>%
  predict(X_test2)

# Keep the biggest probability
predictions <-  apply(class_pred, 1, which.max) - 1 

# Comparison between the real and predicted value
keys_1d <- network_inputs$keys
predicho <- data.frame(predicho = predictions, cod_real = df2$seccion) %>%
  left_join(keys_1d, by =  c("predicho" = "codigo_int")) 

mean(predicho$cod_real == predicho$cod_final)




