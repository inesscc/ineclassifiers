

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
network_inputs <- pre_process_ciuo(df, text_variable =  "glosa_dep_stemm", label =  "cod_final", type = "tfidf")

# Número de clases
clases = length(unique(df$cod_final)) 

# Parámetros de la red y el padding
vocab_size = length(network_inputs$tokenizer$word_index) + 1  
embedding_dim = 300
maxlen = 50

# Arquitectura de la red
input_dim = dim(network_inputs$x_train)[2]
tf$random$set_seed(104) 

model <- keras_model_sequential() 
model %>% 
  layer_dense(90, input_shape = input_dim, activation='relu') %>% 
  layer_dropout(0.5) %>% 
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
  epochs = 12,
  validation_data = list(network_inputs$x_test, network_inputs$y_test),
  batch_size = 128,
  verbose = TRUE
)

# Resultados en el set de entrenamiento y de validación
model %>% 
  evaluate(network_inputs$x_train, network_inputs$y_train)

model %>% 
  evaluate(network_inputs$x_test, network_inputs$y_test)

# Save the model
save_model_hdf5(model, "data/finales/modelos/modelo_red_tfidf_ciuo")
#save_text_tokenizer(network_inputs$tokenizer, filename = "data/finales/modelos/tokenizer_stemm")

# Save the orginal labels
write_feather(network_inputs$keys, path = "data/finales/modelos/keys_ciuo.feather")


#########
# TESTING
#########

# # Recreate the exact same model purely from the file
# model <- load_model_hdf5("data/finales/modelos/modelo_red_tfidf_ciuo")
# tokenizer <- load_text_tokenizer("data/finales/modelos/tokenizer_stemm_ciuo")
# # possible_labels <- read_feather("data/cache/llaves.feather") 
# 
# # # Cargar datos
# path = "data/finales/test.feather"
# df2 = read_feather(path)
# # 
# # Generar la tokenización
# X_test2 = texts_to_matrix(tokenizer, df2$glosa_dep_stemm, mode='tfidf')
# # 
# # # Predecir la clase
# class_pred <- model %>%
#   predict_classes(X_test2)
# 
# predicho <- data.frame(predicho = class_pred, cod_real = df2$cod_final) %>% 
#   left_join(keys %>% rename(cod_pred = cod_final), by = c("predicho" = "codigo_int")) %>% 
#   bind_cols(df2 %>% select(glosa_dep, glosa_dep_stemm, glosa_caenes, idrph, mes))
# 
# mean(predicho$cod_real == predicho$cod_pred)
# 
# source_python("scripts/entrenamiento/calcular_f1.py")
# 
# 
# # Guardar datos
# write_feather(predicho, 'data/finales/predicciones/predict_tfidf_red.feather')




# # Obtener la distribución de probabilidades
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



