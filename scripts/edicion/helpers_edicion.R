#Función que recibe un string y devuelve un string modificado. 
DepurarGlosas <- function(text, stemming = T, borrar_tildes = T) {
  
  
  #text <- iconv(text, sub="")
  
  filtrar   <- stopwords("es") %in% c("estar", "estado", "estados")
  stopwords <-  stopwords("es")[!filtrar]
  text      <- tolower(text)
  text      <- gsub(pattern = "[[:punct:]]", " ", text)
  text      <- gsub(pattern = "[[:digit:]]", " ", text)
  text      <- removeWords(text, stopwords)
  text      <- trimws(text, "both")
  text      <- gsub(pattern = "  ", x = text, replacement = " ")
  
  
  # Borrar tildes
  if (borrar_tildes == T) {
    text      <- gsub(pattern = "á", "a", text)
    text      <- gsub(pattern = "é", "e", text)
    text      <- gsub(pattern = "í", "i", text)
    text      <- gsub(pattern = "ó", "o", text)
    text      <- gsub(pattern = "ú", "u", text)
    
  }
  
  
  #Esta es una opción para hacer el stemming
  if (stemming == T) {
    text <- stemDocument(text, language = "spanish")  
  }
  
  return(text)
}
