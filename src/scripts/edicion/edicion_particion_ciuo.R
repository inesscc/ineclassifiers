#########################
# EDICICIÓN Y PARTICIÓM
#########################

library(tidyverse)
library(readxl)
library(feather)
library(caret)
library(tm)
library(quanteda)

# Cargar helpers
source("scripts/entrenamiento/helpers_edicion.R", encoding = "utf-8")

# Cargar datos codificados
ciuo <- read_feather("data/finales/auditado_ciuo.feather")


# Editar glosa caenes y sacar datos provenientes de la coyuntura ENE
ciuo_edit <- ciuo %>% 
  filter(origen != "ene") %>% # sacar los registros que vienen de la ene
  mutate(glosa_ciuo = paste(b1_1, b1_2)) %>% 
  mutate(glosa_dep_stemm = DepurarGlosas(glosa_ciuo),
         glosa_dep = DepurarGlosas(glosa_ciuo, stemming = F, borrar_tildes = F),
         id = row_number()) %>% 
  mutate(gran_grupo = str_sub(cod_final, 1, 1)  )

# Separar en train y test
set.seed(1234)
trainindex <- createDataPartition(ciuo_edit$cod_final, p=0.8, list=FALSE)
train <- ciuo_edit %>% 
  dplyr::slice(trainindex)

test <- ciuo_edit %>% 
  dplyr::slice(-trainindex)

# Comprobar número de códigos en cada set
length(unique(train$cod_final))
length(unique(test$cod_final))

# Guardar datos
write_feather(train, "data/finales/train_ciuo.feather")
write_feather(test, "data/finales/test_ciuo.feather")

write_csv(train, "data/finales/train_ciuo.csv")
write_csv(test, "data/finales/test_ciuo.csv")


