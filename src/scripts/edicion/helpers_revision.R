find_equal_texts <-  function(false_label, true_label) {
  sentences_wrong <- predicciones %>% 
    filter( cod_pred == false_label & cod_real == true_label) %>%
    mutate(text = as.character(glosa_dep_stemm)) %>% 
    select(text ) 
  
  sentences_wrong_tot <- predicciones %>% 
    filter( cod_real == false_label) %>%
    mutate(text = as.character(glosa_dep_stemm)) %>% 
    select(text) 
  
  col_name <- paste0("cod_", false_label, "_", true_label )
  
  
  equal_sentences <- sentences_wrong %>% 
    inner_join(sentences_wrong_tot %>% distinct(), by = "text") %>% 
    rename(!!parse_expr(col_name) := text)
  
  return(equal_sentences)  
}

percent_equal_texts <- function(false_label, true_label) {
  
  sentences_wrong <- predicciones %>% 
    filter( cod_pred == false_label & cod_real == true_label) %>%
    mutate(text = as.character(glosa_dep_stemm)) %>% 
    select(text ) 
  
  sentences_wrong_tot <- predicciones %>% 
    filter( cod_real == false_label) %>%
    mutate(text = as.character(glosa_dep_stemm)) %>% 
    select(text) 
  
  equal_sentences <- sentences_wrong %>% 
    inner_join(sentences_wrong_tot %>% distinct(), by = "text") 
  
  nrow(equal_sentences) / nrow(sentences_wrong) 
  
}



find_equal_words <- function(false_label, true_label) {
  
  sentences_wrong <- predicciones %>% 
    filter( cod_pred == false_label & cod_real == true_label) %>%
    mutate(text = as.character(glosa_dep_stemm)) %>% 
    select(text ) 
  
  sentences_wrong_tot <- predicciones %>% 
    filter( cod_real == false_label) %>%
    mutate(text = as.character(glosa_dep_stemm)) %>% 
    select(text) 
  
  
  words1 <- sentences_wrong %>% 
    corpus() %>%
    tokens() %>% 
    dfm()
  
  words2 <- sentences_wrong_tot %>% 
    corpus() %>%
    tokens() %>% 
    dfm()
  
  words1_df <- data.frame(word = words1@Dimnames$features)
  words2_df <- data.frame(word = words2@Dimnames$features)
  
  col_name <- paste0("cod_", false_label, "_", true_label )
  
  
  equal_words <- words1_df %>% 
    inner_join(words2_df, by = "word") %>% 
    rename(!!parse_expr(col_name) := word)
  
  return(equal_words)
}

percent_equal_words <- function(false_label, true_label) {
  sentences_wrong <- predicciones %>% 
    filter( cod_pred == false_label & cod_real == true_label) %>%
    mutate(text = as.character(glosa_dep_stemm)) %>% 
    select(text ) 
  
  sentences_wrong_tot <- predicciones %>% 
    filter( cod_real == false_label) %>%
    mutate(text = as.character(glosa_dep_stemm)) %>% 
    select(text) 
  
  
  words1 <- sentences_wrong %>% 
    corpus() %>%
    tokens() %>% 
    dfm()
  
  words2 <- sentences_wrong_tot %>% 
    corpus() %>%
    tokens() %>% 
    dfm()
  
  words1_df <- data.frame(word = words1@Dimnames$features)
  words2_df <- data.frame(word = words2@Dimnames$features)
  
  equal_words <- words1_df %>% 
    inner_join(words2_df, by = "word") 
  
  
  nrow(equal_words) / nrow(words1_df)
  
  
}




