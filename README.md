
El presente documento tiene como objetivo documentar los datos, las
rutinas para entrenar modelos y el funcionamiento de la API, que pone a
disposición los modelos entrenados para clasificar CAENES y CIUO:

## Descripción de los datos

### Variables contenidas en el archivo de CAENES

Variables relevantes para cada uno

This is an R Markdown document. Markdown is a simple formatting syntax
for authoring HTML, PDF, and MS Word documents. For more details on
using R Markdown see <http://rmarkdown.rstudio.com>.

When you click the **Knit** button a document will be generated that
includes both content as well as the output of any embedded R code
chunks within the document. You can embed an R code chunk like this:

## Descripción de las rutinas

Las archivos ubicados en el directorio scripts/entrenamiento contienen
rutinas para llevar a cabo diferentes tipos de entrenamiento. En total,
fueron testeadas 4 metodologías, las cuales consideran diferentes
arquitecturas de redes neuronales:

  - red feed-forward y vectorización de textos mediante TF-IDF

  - red feed-forward y vectorización basada en secuencias

  - red feed-forward y vectorización enriquecida con word embedings

  - red Gated Recurrent Unit (GRU) enriquecida con word embeddings

Dado que los ejercicios consideran CAENES y CIUO, y debido a que fue
necesario entrenar modelos para 1 y 2 dígitos, el número total de
rutinas de entrenamiento es 16: 8 para cada clasificador. Dentro de
dichas 8 rutinas, 4 corresponden a ejercicios realizados a un dígito y 4
a dos dígitos.

Además, el directorio contiene rutinas para la edición de datos y para
la implementación de procesos de entrenamiento y testeo.

A continuación se describe el orden en el que deben ejecutarse los
programas, para reproducir los resultados:

**Entrenamiento CAENES**

1.  edicion\_particion.R: editar textos y dividir el dataset en testeo y
    entrenamiento.

2.  general\_training.R: llevar a cabo el entrenamiento para los 16
    modelos ajustados. Notar que este archivo contiene el entrenamiento
    para los modelos de CIUO y CAENES.

3.  general\_testing.R: llevar a cabo predicciones e identificar el
    rendimiendo en el set de testeo.

**Entrenamiento CIUO**

1.  edicion\_particion\_ciuo.R: editar textos y dividir el dataset en
    testeo y entrenamiento.

2.  general\_training.R: llevar a cabo el entrenamiento para los 16
    modelos ajustados. Notar que este archivo contiene el entrenamiento
    para los modelos de CIUO y CAENES.

3.  general\_testing\_ciuo.R: levar a cabo predicciones e identificar el
    rendimiendo en el set de testeo.

## Descripción del funcionamiento de la API