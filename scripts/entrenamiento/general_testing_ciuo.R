
library(keras)
library(reticulate)
library(tidyverse)
library(caret)
library(feather)
library(reticulate)

##################
# GENERAL INPUTS #
##################

# load helper functions
source("scripts/entrenamiento/helpers_testing.R")

# load testing data
path = "data/finales/test_ciuo.feather"
df = read_feather(path)


# load models
mod_red_seq <- load_model_hdf5("data/finales/modelos/modelo_red_seq_ciuo")
mod_red_seq_1d <- load_model_hdf5("data/finales/modelos/modelo_red_seq_1d_ciuo")

mod_red_tfidf <- load_model_hdf5("data/finales/modelos/modelo_red_tfidf_ciuo")
mod_red_tfidf_1d <- load_model_hdf5("data/finales/modelos/modelo_red_tfidf_1d_ciuo")

mod_red_emb_simple <- load_model_hdf5("data/finales/modelos/modelo_red_emb_simple_ciuo")
mod_red_emb_simple_1d <- load_model_hdf5("data/finales/modelos/modelo_red_emb_simple_1d_ciuo")

mod_red_emb_gru <- load_model_hdf5("data/finales/modelos/modelo_red_emb_gru_ciuo")
mod_red_emb_gru_1d <- load_model_hdf5("data/finales/modelos/modelo_red_emb_gru_1d_ciuo")

# load tokenizers for 1 and 2 digits. Cause stemming was used for some cases, it is necessary load two different tokenizers. In addition, the models for one digit were trained filtering the 999 label. That is why we have 4 different tokenizers. 
tokenizer_stemm <-  load_text_tokenizer(filename = "data/finales/modelos/tokenizer_stemm_ciuo")
tokenizer_stemm_1d <-  load_text_tokenizer(filename = "data/finales/modelos/tokenizer_stemm_1d_ciuo")

tokenizer <-  load_text_tokenizer(filename = "data/finales/modelos/tokenizer_ciuo")
tokenizer_1d <-  load_text_tokenizer(filename = "data/finales/modelos/tokenizer_1d_ciuo")


# load keys. It is neccesary to recover the original label values  
keys <- read_feather("data/finales/modelos/keys_ciuo.feather")
keys_1d <- read_feather("data/finales/modelos/keys_1d_ciuo.feather")




#############
# SEQUENCES #
#############

##### 2 digit #########
seq_2d <- get_accuracy_ciuo(tokenizer_stemm, mod_red_seq, "glosa_dep_stemm", "cod_final", 2, keys ) 
mean(seq_2d$coincidence)

##### 1 digit #########
seq_1d <- get_accuracy_ciuo(tokenizer_stemm_1d, mod_red_seq_1d, "glosa_dep_stemm", "gran_grupo", 1, keys_1d ) 
mean(seq_1d$coincidence)

#########
# TDIDF #
#########

##### 2 digits #########
tfidf_2d <- get_accuracy_ciuo(tokenizer_stemm, mod_red_tfidf, "glosa_dep_stemm", "cod_final", 2, keys) 
mean(tfidf_2d$coincidence)


##### 1 digits #########
tfidf_1d <- get_accuracy_ciuo(tokenizer_stemm_1d, mod_red_tfidf_1d, "glosa_dep_stemm", "gran_grupo", 1, keys_1d) 
mean(tfidf_1d$coincidence)


#####################
# Embeddings simple #
#####################

# 2 digits
emb_simple_2d <- get_accuracy_ciuo(tokenizer, mod_red_emb_simple, "glosa_dep", "cod_final", 2, keys) 
mean(emb_simple_2d$coincidence)

# 1 digit
emb_simple_1d <- get_accuracy_ciuo(tokenizer_1d, mod_red_emb_simple_1d, "glosa_dep", "gran_grupo", 1, keys_1d) 
mean(emb_simple_1d$coincidence)

##################
# Embeddings gru #
##################

# 2 digits
emb_gru_2d <- get_accuracy_ciuo(tokenizer, mod_red_emb_gru, "glosa_dep", "cod_final", 2, keys) 
mean(emb_gru_2d$coincidence)

# 1 digit
emb_gru_1d <- get_accuracy_ciuo(tokenizer_1d, mod_red_emb_gru_1d, "glosa_dep", "seccion", 1, keys_1d) 
mean(emb_gru_1d$coincidence)


############################
# Unir todos los resultados
############################


# We generate a list with the predictions for all methods. The idea is apply the f1 and accuracy for all methods 
predictions <- list(seq_2d, seq_1d, tfidf_2d, tfidf_1d, emb_simple_2d, emb_simple_1d, emb_gru_2d, emb_gru_1d)
names(predictions) <- c("seq_2d", "seq_1d", "tfidf_2d", "tfidf_1d", "emb_simple_2d", "emb_simple_1d", "emb_gru_2d", "emb_gru_1d")

# Cargar función para calcular f1
source_python('scripts/entrenamiento/py_utils.py')

# Apply accuracy and to all predictions 
acc <-  map_dbl(predictions, ~mean(.$coincidence)) %>% 
  as.data.frame() %>% 
  rename(acc = ".")

# Apply  f1 to all predictions 
macro <- map_dbl(predictions, ~get_f1(.$label, .$cod_final, f1_type='macro')) %>% 
  as.data.frame() %>% 
  rename(macro = ".")

micro <- map_dbl(predictions, ~get_f1(.$label, .$cod_final, f1_type='micro')) %>% 
  as.data.frame() %>% 
  rename(micro = ".")

weighted <- map_dbl(predictions, ~get_f1(.$label, .$cod_final, f1_type='weighted')) %>% 
  as.data.frame() %>% 
  rename(weighted = ".")

# Combine all results into one table
results <- bind_cols(acc, macro, micro, weighted)

# save results 
write_feather(results, "data/finales/resultados/results_ciuo.feather")






