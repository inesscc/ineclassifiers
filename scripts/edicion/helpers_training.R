#################
# TRAIN HELPERS #
#################
library(rlang)

pre_process <- function(df, text_variable, label, type = "sequences", maxlen = 40) {
  
  # Si se trabaja a nivel de sección, se sacan los 999, ya que corresponden a glosas que a un dígito probablemente se podrían haber etiquetado
  if (label == "seccion") {
    df <- df %>% 
      filter(cod_final != "999")
  }
  
  # Eliminar las clases que solo tienen un solo ejemplo
  df <- df %>%
    select(all_of(text_variable), cod_final = all_of(label), glosa_caenes) %>% 
    group_by(cod_final) %>% 
    filter(n() >= 2) %>% 
    ungroup() 
  
  
  # Renombrar etiquetas para evitar problemas con el número de categorías 
  keys <- unique(df[["cod_final"]]) %>% 
    as.data.frame() %>% 
    rename(cod_final = ".") %>% 
    arrange(cod_final) %>% 
    mutate(codigo_int = row_number() - 1,
           cod_final = as.character(cod_final))
  
  df <- df %>% 
    left_join(keys, by =  "cod_final") 
  
  # Número de clases
  clases = length(unique(df$codigo_int)) 
  
  # Separar en train y test
  set.seed(1234)
  trainindex <- createDataPartition(df[["cod_final"]], p=0.9,list=FALSE)
  
  train <- df %>% 
    dplyr::slice(trainindex)
  
  test <- df %>% 
    dplyr::slice(-trainindex)
  
  # Pasar todo a formato array
  y_train <- as.array(train$codigo_int)
  y_test <- as.array(test$codigo_int)
  x_train <- as.array(train[[text_variable]])
  x_test <- as.array(test[[text_variable]])
  
  # Tokenizar
  tokenizer <-  text_tokenizer( oov_token = "OOV")
  fit_text_tokenizer(tokenizer, x_train)  
  
  #save_text_tokenizer(tokenizer, "data/finales/modelos/modelo_red_seq")
  
  if (type == "sequences") {
    # Transformar en secuencias
    train_sequences = texts_to_sequences(tokenizer, x_train)
    test_sequences = texts_to_sequences(tokenizer, x_test)
    

    
    # Generar padding
    x_train <- pad_sequences(train_sequences, padding='post', maxlen=maxlen)
    x_test <- pad_sequences(test_sequences, padding='post', maxlen=maxlen)

  } else if (type == "tfidf") {
    
    # Generar tfidf
    x_train = texts_to_matrix(tokenizer, x_train, mode='tfidf')
    x_test = texts_to_matrix(tokenizer, x_test, mode = "tfidf")  
  }
  
    
  #save_text_tokenizer(tokenizer, filename = "scripts/entrenamiento/app/data/tokenizer_red_tfidf")
  
  output <- list(x_train, y_train,  x_test, y_test, tokenizer, keys)
  names(output) <- c("x_train", "y_train",  "x_test", "y_test", "tokenizer", "keys")
  return(output)
  
}



pre_process_ciuo <- function(df, text_variable, label, type = "sequences", maxlen = 40) {
  
  # Si se trabaja a nivel de sección, se sacan los 999, ya que corresponden a glosas que a un dígito probablemente se podrían haber etiquetado
  if (label == "gran_grupo") {
    df <- df %>% 
      filter(cod_final != "999")
  }
  
  # Eliminar las clases que solo tienen un solo ejemplo
  df <- df %>%
    select(all_of(text_variable), cod_final = all_of(label), glosa_ciuo) %>% 
    group_by(cod_final) %>% 
    filter(n() >= 2) %>% 
    ungroup() 
  
  
  # Renombrar etiquetas para evitar problemas con el número de categorías 
  keys <- unique(df[["cod_final"]]) %>% 
    as.data.frame() %>% 
    rename(cod_final = ".") %>% 
    arrange(cod_final) %>% 
    mutate(codigo_int = row_number() - 1,
           cod_final = as.character(cod_final))
  
  df <- df %>% 
    left_join(keys, by =  "cod_final") 
  
  # Número de clases
  clases = length(unique(df$codigo_int)) 
  
  # Separar en train y test
  set.seed(1234)
  trainindex <- createDataPartition(df[["cod_final"]], p=0.9,list=FALSE)
  
  train <- df %>% 
    dplyr::slice(trainindex)
  
  test <- df %>% 
    dplyr::slice(-trainindex)
  
  # Pasar todo a formato array
  y_train <- as.array(train$codigo_int)
  y_test <- as.array(test$codigo_int)
  x_train <- as.array(train[[text_variable]])
  x_test <- as.array(test[[text_variable]])
  
  # Tokenizar
  tokenizer <-  text_tokenizer( oov_token = "OOV")
  fit_text_tokenizer(tokenizer, x_train)  
  
  #save_text_tokenizer(tokenizer, "data/finales/modelos/modelo_red_seq")
  
  if (type == "sequences") {
    # Transformar en secuencias
    train_sequences = texts_to_sequences(tokenizer, x_train)
    test_sequences = texts_to_sequences(tokenizer, x_test)
    
    
    
    # Generar padding
    x_train <- pad_sequences(train_sequences, padding='post', maxlen=maxlen)
    x_test <- pad_sequences(test_sequences, padding='post', maxlen=maxlen)
    
  } else if (type == "tfidf") {
    
    # Generar tfidf
    x_train = texts_to_matrix(tokenizer, x_train, mode='tfidf')
    x_test = texts_to_matrix(tokenizer, x_test, mode = "tfidf")  
  }
  
  
  #save_text_tokenizer(tokenizer, filename = "scripts/entrenamiento/app/data/tokenizer_red_tfidf")
  
  output <- list(x_train, y_train,  x_test, y_test, tokenizer, keys)
  names(output) <- c("x_train", "y_train",  "x_test", "y_test", "tokenizer", "keys")
  return(output)
  
}
