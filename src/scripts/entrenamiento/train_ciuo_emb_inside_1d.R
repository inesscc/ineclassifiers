

library(keras)
library(reticulate)
library(tidyverse)
library(caret)
library(feather)

# Cargar funciones
source("scripts/entrenamiento/helpers_edicion.R", encoding = "utf-8")
source("scripts/entrenamiento/helpers_training.R", encoding = "utf-8")

# Cargar datos
path = "data/finales/train_ciuo.feather"
df = read_feather(path)

network_inputs <- pre_process_ciuo(df, text_variable =  "glosa_dep", label =  "gran_grupo", type = "sequences")


# Parámetros de la red y el padding
vocab_size = length(network_inputs$tokenizer$word_index) + 1  
embedding_dim = 300
maxlen = 40
tokenizer <- network_inputs$tokenizer
clases = length(unique(network_inputs$y_train))


# Crear los embeddings a partir del modelo gensim. Tarda en correr, debido a que el modelo es muy pesado
source_python("scripts/entrenamiento/red/create_embeddings.py")

# Matriz de embeddings creada con código python
embedding_matrix <-  py$embedding_matrix
dim(embedding_matrix)

tf$random$set_seed(104) 
model <- keras_model_sequential()
model %>%
  layer_embedding(input_dim=vocab_size,
                  output_dim=300,
                  input_length=maxlen,
                  weights = list(embedding_matrix),
                  trainable = F) %>%
  layer_global_average_pooling_1d() %>%
  #layer_global_max_pooling_1d() %>%
  layer_dropout(0.4) %>%
  layer_dense(1000, activation='relu') %>%
  layer_dropout(0.4) %>%
  layer_dense(clases, activation='softmax')


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
  epochs = 120,
  validation_data = list(network_inputs$x_test, network_inputs$y_test),
  batch_size = 256,
  verbose = TRUE
)

save_model_hdf5(model, "data/finales/modelos/modelo_red_emb_simple_1d_ciuo")
save_text_tokenizer(network_inputs$tokenizer, filename = "data/finales/modelos/tokenizer_1d_ciuo")



#########
# TESTING
#########

# Recreate the exact same model purely from the file
# 
# 
# # Generar la tokenización
# X_test2 = texts_to_sequences(tokenizer, df2$glosa_dep)
# # 
# # # Hacer padding para que llenar con ceros e igualar el largo de los vectores
# maxlen = 40
# X_test2 = pad_sequences(X_test2, padding='post', maxlen=maxlen)
# # 
# # # Predecir la clase
# class_pred <- model %>%
#   predict_classes(X_test2)
# # 
# predicho <- data.frame(predicho = class_pred, cod_real = df2$seccion) %>%
#   left_join(keys_1d, by =  c("predicho" = "codigo_int")) 
# # 
# # 
# mean(predicho$cod_real == predicho$cod_final)
# 
# # Guardar
# write_feather(predicho, 'data/finales/predicciones/predict_emb_inside_red_1d.feather')



# Obtener la distribución de probabilidades
# probs <- model %>%
#   predict(X_test2)
# 
# # Sacar las 3 probabilidades más altas y los códigos asociados
# mas_altas <-  as.data.frame(matrix(nrow = nrow(probs), ncol = 4))
# codigos <- as.data.frame(matrix(nrow = nrow(probs), ncol = 4))
# 
# mas_altas <- apply(probs, 1, extraer_probs, n = 4) %>% 
#   t()
# 
# for (row in 1:nrow(mas_altas)) { #
#   for (col in 1:ncol(mas_altas)) {
#     codigos[row, col] <-  which( probs[row, ] == mas_altas[row, col ] )    
#   }
# }
# 
# codigos <- codigos - 1 # para homologar con el sistema de índices de python
# 
# 
# # Pegar los códigos originales 
# codigos_probables <- imap(codigos, ~codigos %>% 
#                             rename(codigo_int = .y) %>% 
#                             select(codigo_int) %>% 
#                             left_join(possible_labels, 
#                                       by = "codigo_int") %>% 
#                             select(-codigo_int)) %>% 
#   reduce(bind_cols)
# 
# names(codigos_probables) <- str_remove(names(codigos_probables), pattern = "\\...")
# 
# # Crear tabla que contiene los 4 códigos más probables y el código real
# final <- bind_cols(codigos_probables, real = df2$codigo, glosa = df2$glosa, id = df2$id) %>% 
#   mutate(coincide3 = if_else(codigo1 == real | codigo2 == real | codigo3 == real, 1, 0),
#          coincide4 = if_else(codigo1 == real | codigo2 == real | codigo3 == real | codigo4 == real, 1, 0)) %>% 
#   bind_cols(as.data.frame(mas_altas)) 
# 
# 
# mean(final$codigo1 == final$real)
# mean(final$coincide3)
# mean(final$coincide4)
# 


