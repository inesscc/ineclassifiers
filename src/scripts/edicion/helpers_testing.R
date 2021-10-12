###################
# HELPERS TESTING #
###################

# 
# digit = 1
# token = tokenizer_1d
# model <- mod_red_emb_gru_1d
# var_x = "glosa_dep"
# var_y <- "seccion"
# keys <- keys_1d

get_accuracy <- function(token, model, var_x, var_y, digit, keys) {
  
  # Capture kind of model into a string in order to apply different processing for tfidf 
  representation <- as_string(enexpr(model))
  
  if (digit == 1) {
    
    # Vector format
    x_test <-  df %>% 
      filter(!is.na(seccion)) %>% 
      pull(parse_expr(var_x))
    
    y_test <- df %>% 
      filter(!is.na(seccion)) %>% 
      pull(seccion)
    
  }  else {
    # Array format
    x_test <- as.array(df[[var_x]])
    y_test <- as.array(df[[var_y]])

    
  }
  

  #### This is the case for sequences

  if (str_detect(representation, pattern = "seq") | str_detect(representation, pattern = "emb_simple") | str_detect(representation, pattern = "gru") ) {
    
    # Tokenize using file saved during the training 
    test_sequences = texts_to_sequences(token, x_test)
    
    #  Padding
    maxlen = 40
    x_test <- pad_sequences(test_sequences, padding = 'post', maxlen=maxlen) 
    
  ### This is the case for tfidf    
  } else {
    x_test = texts_to_matrix(token, x_test,  mode='tfidf')
    
  }
    

  # Prediction for the normal case
  if (str_detect(representation, pattern = "gru") == F ) {
    predict <- predict_classes(model, x_test)    
  # gru case
  } else {
    
    # Probability distribution for all classes
    class_pred <- model %>%
      predict(x_test)
    
    # Keep the biggest probability
    predict <-  apply(class_pred, 1, which.max) - 1 
      
  }

  
  # Recover the original label with the keys file. Also, we add the test label
  predictions <- as.data.frame(predict) %>% 
    rename(codigo_int = predict) %>% 
    left_join(keys, by = "codigo_int") %>% 
    bind_cols(label = y_test) 
  
  # Check accuracy
  results <- predictions %>% 
    mutate(coincidence = if_else(label == cod_final, 1, 0)) 
  
  return(results)
  
}



get_accuracy_ciuo <- function(token, model, var_x, var_y, digit, keys) {
  
  # Capture kind of model into a string in order to apply different processing for tfidf 
  representation <- as_string(enexpr(model))
  
  if (digit == 1) {
    
    # Vector format
    x_test <-  df %>% 
      filter(!is.na(gran_grupo)) %>% 
      pull(parse_expr(var_x))
    
    y_test <- df %>% 
      filter(!is.na(gran_grupo)) %>% 
      pull(gran_grupo)
    
  }  else {
    # Array format
    x_test <- as.array(df[[var_x]])
    y_test <- as.array(df[[var_y]])
    
    
  }
  
  
  #### This is the case for sequences
  
  if (str_detect(representation, pattern = "seq") | str_detect(representation, pattern = "emb_simple") | str_detect(representation, pattern = "gru") ) {
    
    # Tokenize using file saved during the training 
    test_sequences = texts_to_sequences(token, x_test)
    
    #  Padding
    maxlen = 40
    x_test <- pad_sequences(test_sequences, padding = 'post', maxlen=maxlen) 
    
    ### This is the case for tfidf    
  } else {
    x_test = texts_to_matrix(token, x_test,  mode='tfidf')
    
  }
  
  
  # Prediction for the normal case
  if (str_detect(representation, pattern = "gru") == F ) {
    
    predict <- predict_classes(model, x_test)    
    #return("perro")
    
    # gru case
  } else {
    
    # Probability distribution for all classes
    class_pred <- model %>%
      predict(x_test)
    
    # Keep the biggest probability
    predict <-  apply(class_pred, 1, which.max) - 1 
    
  }
  
  
  # Recover the original label with the keys file. Also, we add the test label
  predictions <- as.data.frame(predict) %>%
    rename(codigo_int = predict) %>%
    left_join(keys, by = "codigo_int") %>% 
    bind_cols(label = y_test) 
  
  
  
  # Check accuracy
  results <- predictions %>% 
    mutate(coincidence = if_else(label == cod_final, 1, 0)) 
  #return(list(predictions, keys, predictions, results))
  
  return(results)
  
}

