# CI11771 - Representação

* Extrai as características de um conjunto de imagens utilizando HOG (Histogram of Oriented Gradients) e salva em um arquivo de saída.

## Execução

* *python3 feat_extract.py <diretório de entrada> <arquivo de saída>*

## Saída

* Um arquivo texto com o nome passado como parâmetro contendo as características de cada imagem do diretório de entrada no formato:

```
<classe da imagem 1> <vetor de características 1>
<classe da imagem 2> <vetor de características 2>
...
<classe da imagem n> <vetor de características n>

```

## Autor
* Gabriel de Oliveira Pontarolo, GRR20203895, gop20