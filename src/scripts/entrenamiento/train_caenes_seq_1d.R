
##############################
# Entrenar red con secuencias#
##############################

library(keras)
library(reticulate)
library(tidyverse)
library(caret)
library(feather)
library(tensorflow)


# Cargar funciones
source("scripts/entrenamiento/helpers_edicion.R", encoding = "utf-8")

# Cargar datos
path = "data/finales/train.feather"
df <-  read_feather(path)

# Eliminar las clases que solo tienen un solo ejemplo o que tienen código 999, ya que fue puesto con una lógica de 2 dígitos
df <- df %>% 
  select(glosa_dep_stemm, cod_final, glosa_caenes, variable, levantamiento, origen, cise, seccion) %>% 
  group_by(seccion) %>% 
  filter(n() >= 2 & cod_final != "999") %>% 
  ungroup()

# Renombrar etiquetas para evitar problemas con el número de categorías 
keys <- unique(df$seccion) %>% 
  as.data.frame() %>% 
  rename(seccion = ".") %>% 
  arrange(seccion) %>% 
  mutate(seccion_int = row_number() - 1,
         seccion = as.character(seccion))

df <- df %>% 
  left_join(keys, by = "seccion")

# Número de clases
clases = length(unique(df$seccion_int)) 

# Separar en train y test
set.seed(1234)
trainindex <- createDataPartition(df$seccion_int, p=0.9, list=FALSE)
train <- df %>% 
  dplyr::slice(trainindex)

test <- df %>% 
  dplyr::slice(-trainindex)

# Pasar todo a formato array
sentences_train <- as.array(train$glosa_dep_stemm)
sentences_test <- as.array(test$glosa_dep_stemm)
y_test <- as.array(test$seccion_int)
y_train <- as.array(train$seccion_int)


# Tokenizar
tokenizer <-  text_tokenizer( oov_token = "OOV")
fit_text_tokenizer(tokenizer, sentences_train)



# Transformar en secuencias
X_train = texts_to_sequences(tokenizer, sentences_train)
X_test = texts_to_sequences(tokenizer, sentences_test)

print(sentences_train[10])
print(X_train[10])


# Parámetros de la red y el padding
vocab_size = length(tokenizer$word_index) + 1  
embedding_dim = 300
maxlen = 40

# Generar padding
X_train <- pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test <- pad_sequences(X_test, padding='post', maxlen=maxlen)

print(X_train[12, ])

# Arquitectura de la red
tf$random$set_seed(104) 
model <- keras_model_sequential()
model %>%
  layer_embedding(input_dim=vocab_size,
                  output_dim=embedding_dim,
                  input_length=maxlen) %>%
  layer_global_max_pooling_1d() %>%
  layer_dropout(0.5) %>%
  layer_dense(125, activation='relu') %>%
  layer_dropout(0.3) %>%
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
  X_train, y_train,
  epochs = 15,
  validation_data = list(X_test, y_test),
  batch_size = 256,
  verbose = TRUE
)


# Generar una predicción para el set de validación
class_pred_val <- model %>% 
  predict_classes(X_test)


# Resultados en el set de entrenamiento y de validación
model %>% 
  evaluate(X_train, y_train)

model %>% 
  evaluate(X_test, y_test)

# Save the model
save_model_hdf5(model, "data/finales/modelos/modelo_red_seq_1d")
save_text_tokenizer(tokenizer, filename = "data/finales/modelos/tokenizer_stemm_1d")

# Guardar llaves para la predicción
write_feather(keys, "data/finales/modelos/keys_1d.feather")



#########
# TESTING
#########

# Recreate the exact same model purely from the file
# model <- load_model_hdf5("data/finales/modelos/modelo_red_seq_1d")
# tokenizer <- load_text_tokenizer("data/finales/modelos/tokenizer_stemm_1d")
# #possible_labels <- read_feather("data/cache/llaves.feather")
# 
# # # Cargar datos
# path = "data/finales/test.feather"
# df2 = read_feather(path)
# 
# # # Sacar casos con 999
# df2 <- df2 %>%
#   filter(cod_final != "999")
# # 
# # # Generar la tokenización
# X_test2 = texts_to_sequences(tokenizer, df2$glosa_dep_stemm)
# # 
# # # Hacer padding para que llenar con ceros e igualar el largo de los vectores
# maxlen = 40
# X_test2 = pad_sequences(X_test2, padding='post', maxlen=maxlen)
# # 
# # # Predecir la clase
# class_pred <- model %>%
#   predict_classes(X_test2)
# 
# predicho <- data.frame(predicho = class_pred, cod_real = df2$seccion) %>% 
#   left_join(keys %>%  rename(cod_predicho = seccion), by =  c("predicho" = "seccion_int")) %>% 
#   bind_cols(df2 %>% select(glosa_dep, glosa_caenes, idrph, mes))
# 
# 
# mean(predicho$cod_real == predicho$cod_pred)
# 
# 
# write_feather(predicho, 'data/finales/predicciones/predict_seq_red_1d.feather')
# 



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



