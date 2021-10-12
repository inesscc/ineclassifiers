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
caenes <- read_feather("data/finales/auditado_ene_long.feather")

# Cargar datos de 2018
caenes_2018 <- read_feather("data/finales/datos_2018.feather")


# Cargar clasificador caenes
clasificador <- read_excel("data/finales/clasificador_caenes.xlsx")
names(clasificador) <- tolower(names(clasificador))
names(clasificador) <- str_replace(pattern = "ó", string = names(clasificador), replacement = "o")


# Editar glosa caenes y sacar datos provenientes de la coyuntura ENE
caenes_edit <- caenes %>% 
  filter(origen != "ene") %>% # sacar los registros que vienen de la ene
  mutate(glosa_dep_stemm = DepurarGlosas(glosa_caenes),
         glosa_dep = DepurarGlosas(glosa_caenes, stemming = F, borrar_tildes = F),
         glosa_dep_stemm = paste(glosa_dep_stemm, cise),
         id = row_number()) 

# Editar glosa caenes para los datos de 2018
caenes_edit_2018 <- caenes_2018 %>% 
  mutate(glosa_dep_stemm = DepurarGlosas(glosa_caenes),
         glosa_dep = DepurarGlosas(glosa_caenes, stemming = F, borrar_tildes = F),
         glosa_dep_stemm = paste(glosa_dep_stemm, cise)) 

# Pegar código de la división
caenes_edit <- caenes_edit %>% 
  left_join(clasificador %>% select(seccion, division), by = c("cod_final" = "division"))

# Separar en train y test
set.seed(1234)
trainindex <- createDataPartition(caenes_edit$cod_final, p=0.8, list=FALSE)
train <- caenes_edit %>% 
  dplyr::slice(trainindex)

test <- caenes_edit %>% 
  dplyr::slice(-trainindex)

# Comprobar número de códigos en cada set
length(unique(train$cod_final))
length(unique(test$cod_final))

# Guardar datos
write_feather(train, "data/finales/train.feather")
write_feather(test, "data/finales/test.feather")
write_feather(caenes_edit_2018, "data/finales/train_2018.feather")

write_csv(train, "data/finales/train.csv")
write_csv(test, "data/finales/test.csv")


x <- train %>% 
  mutate(id = paste0(idrph, mes, variable)) %>% 
  group_by(id) %>% 
  mutate(contar = n()) %>% 
  ungroup() %>% 
  arrange(contar)

x %>% filter(is.na(variable)) %>% View()

sum(is.na(x$id))
length(unique(x$id))


