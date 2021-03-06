---
authors:
- Gian Maurício Fritsche
title: |
    Reconhecimento de Padrões\
    Trabalho Final
...

Descrição do trabalho
=====================

Para a realização deste trabalho foi utilizada a base de imagens
SIMPSONS[^1]. O objetivo é construir um sistema de reconhecimento de
padrões que discriMínimoe as cinco classes (representando os cinco
personagens principais: Bart, Homer, Lisa, Maggie e Marge).

[b]<span>0.15</span> ![Bart](report/bart001 "fig:") [fig:bart]

 

[b]<span>0.15</span> ![Homer](report/homer001 "fig:") [fig:homer]

 

[b]<span>0.15</span> ![Lisa](report/lisa001 "fig:") [fig:lisa]

 

[b]<span>0.15</span> ![Maggie](report/maggie001 "fig:") [fig:maggie]

 

[b]<span>0.15</span> ![Marge](report/marge001 "fig:") [fig:marge]

[fig:exemplos]

Na Figura [fig:exemplos] são apresentados exemplos de imagens para cada
uma das cinco classes. Foram utilizados três métodos para extração de
características: Histograma de cor, Momentos de Hu e Histograma da
orientação dos gradientes <span>*Histogram of oriented gradients*</span>
(HOG). Inicialmente toda imagem recebida pelo módulo de extração é
redimensionada para $150 \times 150$, em seguida é enviada para o método
de extração de características selecionado.

Extração de características
---------------------------

Para o histograma de cor, cada canal (cor de 0 à 255) foi dividida em
quatro partes (bins) e calculado quantos pixels se encaixam em cada
parte (para cada cor). Retornando assim um vetor de características com
64 posições. A quantidade de divisões (bins) é um parâmetro do método,
porém apenas o valor quatro foi avaliado. Outro método de extração de
características utilizado foi o Momentos de Hu, que são invariantes a
translação, rotação e escala. Para sua utilização a imagem foi
convertida para tons de cinza. Este método retorna um vetor de sete
características (os sete momentos de Hu). É sugerido que este método
seja utilizado após a segmentação da imagem, porém esta etapa não foi
realizada. O terceiro método de extração de características utilizado
foi o HOG, que utiliza os gradientes da imagem para capturar contornos,
silhuetas e algumas informações de textura.

Classificação
-------------

Para a classificação das imagens, inicialmente os vetores de
característica são normalizados. Em seguida é aplicado um dos três
classificadores implementados: <span>*Linear DiscriMínimoant
Analysis*</span> (LDA), <span>*K-th Nearest Neighbor*</span> (KNN) e
<span>*Support Vector Machine *</span> (SVM). O método KNN, apresenta
dois parâmetros, o número de vizinhos ($k$) e a métrica de distância. Os
valores utilizados foram $k=5$ e distância euclidiana. Não foram
avaliados outros valores. Para o SVM o modelo foi aprendido por meio de
<span>*GridSearch*</span>.

Fusão
-----

Para a fusão foram construídas duas lista, uma com os métodos de
extração de características (<span>*features*</span>) e outra com os
classificadores (<span>*clasifiers*</span>). Então para cada combinação
$[feature, Classificador]$ é construído e treinado um classificador. Em
seguida os exemplos de teste são classificados utilizando todos os
classificadores construídos. Para a fusão da saída dos classificadores
foram implementados cinco métodos: soma, mínimo, máximo, produto e
<span>*Borda count*</span>. Os quatro primeiros utilizam as
probabilidades retornadas por cada classificador, enquanto para o
<span>*Borda count*</span> foi calculado os rankings (a partir das
probabilidades).

Análise da classificação
========================

Inicialmente foram avaliados todas as combinações de métodos de extração
de características e de classificação. Assim, foram instanciados nove
classificadores e os resultados foram agrupados por método de extração
de características. Os resultados são apresentados nas Tabelas
[tbl:colorhistogram], [tbl:hog] e [tbl:humoments]. A escolha dos métodos
de extração de características apresentaram um maior impacto no acerto
do que os métodos de classificação. Além disso, todos os métodos
apresentaram <span>*bias*</span> para o personagem `bart`, provavelmente
por ser a classe que apresenta o maior número de exemplos (tanto no
teste quanto na validação).

Para análise do desempenho dos classificadores é apresentado o acerto
geral, o acerto para cada classe e também o acerto médio nas cinco
classes. Esta análise permite identificar classificadores que tenham um
bom desempenho geral, porém tendem a acertar uma (ou poucas) classes (as
que tenham mais exemplos). Por exemplo, a combinação Histograma de cor e
SVM apresentou melhor desempenho geral e pior desempenho médio do que a
combinação Histograma de cor e KNN.

[!htb] [tbl:colorhistogram]

<span>l|l|l|l|l|l|l</span>\
\
& bart & homer & lisa & maggie & marge & Acerto:\
bart & 27 & 7 & 1 & 0 & 0 & 77.14%\
homer & 5 & 15 & 3 & 1 & 1 & 60.00%\
lisa & 4 & 3 & 6 & 0 & 0 & 46.15%\
maggie & 1 & 3 & 0 & 8 & 0 & 66.67%\
marge & 5 & 1 & 1 & 0 & 3 & 30.00%\
& & & **** & **** & **media** & **55.99%**\
\
\
& bart & homer & lisa & maggie & marge & Acerto:\
bart & 30 & 4 & 1 & 0 & 0 & 85.71%\
homer & 2 & 16 & 6 & 0 & 1 & 64.00%\
lisa & 2 & 3 & 8 & 0 & 0 & 61.54%\
maggie & 2 & 2 & 0 & 8 & 0 & 66.67%\
marge & 3 & 1 & 3 & 0 & 3 & 30.00%\
& & & & & **media** & **61.58%**\
\
\
& bart & homer & lisa & maggie & marge & Acerto:\
bart & 32 & 2 & 1 & 0 & 0 & 91.43%\
homer & 7 & 12 & 5 & 1 & 0 & 48.00%\
lisa & 3 & 1 & 8 & 1 & 0 & 61.54%\
maggie & 4 & 1 & 0 & 7 & 0 & 58.33%\
marge & 7 & 0 & 0 & 0 & 3 & 30.00%\
& & & & **** & **media** & **57.86%**\

O método Histograma de cor (Tabela [tbl:colorhistogram]) apresentou
melhores resultados do que os demais métodos de extração de
características. Tanto em termos de acerto geral, quanto em termos de
acerto médio entre as cinco classes. A escolha do método de
classificação apresentou menor impacto no desempenho, com pequena
vantagem para o LDA (no caso de Histograma de cor).

[!htb] [tbl:hog]

<span>l|l|l|l|l|l|l</span>\
\
& bart & homer & lisa & maggie & marge & Acerto:\
bart & 26 & 9 & 0 & 0 & 0 & 74.29%\
homer & 14 & 9 & 1 & 0 & 1 & 36.00%\
lisa & 5 & 5 & 3 & 0 & 0 & 23.08%\
maggie & 3 & 5 & 1 & 2 & 1 & 16.67%\
marge & 5 & 3 & 1 & 0 & 1 & 10.00%\
& & & & & **media** & **32.01%**\
\
\
& bart & homer & lisa & maggie & marge & Acerto:\
bart & 26 & 6 & 2 & 1 & 0 & 74.29%\
homer & 11 & 9 & 0 & 2 & 3 & 36.00%\
lisa & 9 & 3 & 1 & 0 & 0 & 7.69%\
maggie & 6 & 2 & 0 & 4 & 0 & 33.33%\
marge & 6 & 2 & 1 & 0 & 1 & 10.00%\
& & & & & **media** & **32.26%**\
\
\
& bart & homer & lisa & maggie & marge & Acerto:\
bart & 32 & 3 & 0 & 0 & 0 & 91.43%\
homer & 11 & 14 & 0 & 0 & 0 & 56.00%\
lisa & 8 & 4 & 1 & 0 & 0 & 7.69%\
maggie & 9 & 3 & 0 & 0 & 0 & 0.00%\
marge & 7 & 3 & 0 & 0 & 0 & 0.00%\
& & & & & **media** & **31.02%**\
\

[!htb] [tbl:humoments]

<span>l|l|l|l|l|l|l</span>\
\
& bart & homer & lisa & maggie & marge & Acerto:\
bart & 28 & 7 & 0 & 0 & 0 & 80.00%\
homer & 14 & 9 & 0 & 2 & 0 & 36.00%\
lisa & 11 & 2 & 0 & 0 & 0 & 0.00%\
maggie & 8 & 3 & 0 & 0 & 1 & 0.00%\
marge & 3 & 2 & 0 & 0 & 5 & 50.00%\
**** & **** & **** & **** & **** & **media** & **33.20%**\
\
\
& bart & homer & lisa & maggie & marge & Acerto:\
bart & 25 & 6 & 0 & 1 & 3 & 71.43%\
homer & 15 & 7 & 0 & 3 & 0 & 28.00%\
lisa & 11 & 1 & 0 & 0 & 1 & 0.00%\
maggie & 8 & 1 & 0 & 1 & 2 & 8.33%\
marge & 3 & 1 & 0 & 0 & 6 & 60.00%\
**** & **** & **** & **** & **** & **media** & **33.55%**\
\
\
& bart & homer & lisa & maggie & marge & Acerto:\
bart & 20 & 6 & 6 & 1 & 2 & 57.14%\
homer & 10 & 8 & 3 & 3 & 1 & 32.00%\
lisa & 10 & 0 & 0 & 2 & 1 & 0.00%\
maggie & 4 & 2 & 1 & 3 & 2 & 25.00%\
marge & 2 & 1 & 1 & 1 & 5 & 50.00%\
**** & **** & **** & **** & **** & **media** & **32.83%**\

Nos experimentos realizados com Histograma de orientação dos gradientes
(HOG) (Tabela [tbl:hog]) o desempenho foi relativamente mais baixo
($49.47\%$ no melhor caso [KNN]). Principalmente em termos de desempenho
médio. Neste caso a diferença entre o acerto para uma classe e outra
chega a ser de $91.43\%$ (KNN). Sendo que, em geral, poucos exemplos
foram classificados nas duas últimas classes (as que apresentam menos
exemplos).

Utilizando o método de extração de características Momentos de Hu
(Tabela [tbl:humoments]) o desempenho geral também foi ruim. Porém, nos
três casos (SVM, LDA e KNN) apresentou melhor desempenho médio do que
quando utilizado HOG.

[!htb] [tbl:fusionall]

<span>l|l|l|l|l|l|l</span>\
\
& bart & homer & lisa & maggie & marge & Acerto:\
bart & 34 & 1 & 0 & 0 & 0 & 97.14%\
homer & 13 & 12 & 0 & 0 & 0 & 48.00%\
lisa & 11 & 1 & 1 & 0 & 0 & 7.69%\
maggie & 7 & 3 & 0 & 2 & 0 & 16.67%\
marge & 8 & 0 & 0 & 0 & 2 & 20.00%\
**** & **** & **** & **** & **** & **media** & **37.90%**\
\
\
& bart & homer & lisa & maggie & marge & Acerto:\
bart & 33 & 2 & 0 & 0 & 0 & 94.29%\
homer & 7 & 17 & 1 & 0 & 0 & 68.00%\
lisa & 8 & 3 & 2 & 0 & 0 & 15.38%\
maggie & 2 & 2 & 0 & 8 & 0 & 66.67%\
marge & 4 & 1 & 1 & 0 & 4 & 40.00%\
**** & **** & **** & **** & **** & **media** & **56.87%**\
\
\
& bart & homer & lisa & maggie & marge & Acerto:\
bart & 29 & 3 & 2 & 1 & 0 & 82.86%\
homer & 6 & 12 & 6 & 0 & 1 & 48.00%\
lisa & 3 & 3 & 7 & 0 & 0 & 53.85%\
maggie & 2 & 2 & 0 & 8 & 0 & 66.67%\
marge & 3 & 1 & 2 & 0 & 4 & 40.00%\
**** & **** & **** & **** & **** & **media** & **58.27%**\
\
\
& bart & homer & lisa & maggie & marge & Acerto:\
bart & 31 & 3 & 0 & 1 & 0 & 88.57%\
homer & 14 & 11 & 0 & 0 & 0 & 44.00%\
lisa & 12 & 1 & 0 & 0 & 0 & 0.00%\
maggie & 7 & 1 & 1 & 3 & 0 & 25.00%\
marge & 9 & 1 & 0 & 0 & 0 & 0.00%\
**** & **** & **** & **** & **** & **media** & **31.51%**\
\
\
& bart & homer & lisa & maggie & marge & Acerto:\
bart & 32 & 3 & 0 & 0 & 0 & 91.43%\
homer & 15 & 10 & 0 & 0 & 0 & 40.00%\
lisa & 12 & 1 & 0 & 0 & 0 & 0.00%\
maggie & 7 & 1 & 1 & 3 & 0 & 25.00%\
marge & 9 & 1 & 0 & 0 & 0 & 0.00%\
**** & **** & **** & **** & **** & **media** & **31.29%**\

Foi avaliada a fusão dos nove classificadores (três métodos de extração
de características combinados com três métodos de classificação) com
cinco diferentes métodos de fusão. Os métodos de fusão mais rígidos,
(produto e mínimo) foram os que apresentaram o pior desempenho,
principalmente quanto ao desempenho médio entre as cinco classes. Sendo
pior do que a maioria dos nove classificadores aplicados
individualmente. O método <span>*Borda count*</span> apresentou um
desempenho mediano porém pior do que todos os classificadores baseados
em histograma de cor individualmente. O método Máximo e Soma
apresentaram desempenho mais próximo aos melhores aplicados
individualmente. Porém, ainda assim, menores do que o classificador LDA
com Histograma de cor aplicado individualmente. Este classificador
obteve um desempenho de $68.42\%$ no geral e $61.58\%$ na média entre as
classes. Sendo que o melhor desempenho geral obtido por meio de fusão de
classificadores foi com o método soma ($67.37\%$) e o melhor desempenho
médio pelo método Máximo ($58.27\%$).

[!htb] [tbl:huandcolor]

<span>l|l|l|l|l|l|l</span>\
\
& bart & homer & lisa & maggie & marge & Acerto:\
bart & 32 & 2 & 1 & 0 & 0 & 91.43%\
homer & 10 & 12 & 1 & 2 & 0 & 48.00%\
lisa & 13 & 0 & 0 & 0 & 0 & 0.00%\
maggie & 5 & 2 & 0 & 5 & 0 & 41.67%\
marge & 6 & 1 & 0 & 0 & 3 & 30.00%\
**** & **** & **** & **** & **** & **media** & **42.22%**\
\
\
& bart & homer & lisa & maggie & marge & Acerto:\
bart & 31 & 3 & 1 & 0 & 0 & 88.57%\
homer & 5 & 14 & 4 & 1 & 1 & 56.00%\
lisa & 4 & 2 & 7 & 0 & 0 & 53.85%\
maggie & 2 & 2 & 0 & 8 & 0 & 66.67%\
marge & 4 & 1 & 1 & 0 & 4 & 40.00%\
**** & **** & **** & **** & **** & **media** & **61.02%**\
\
\
& bart & homer & lisa & maggie & marge & Acerto:\
bart & 30 & 4 & 1 & 0 & 0 & 85.71%\
homer & 3 & 15 & 6 & 0 & 1 & 60.00%\
lisa & 2 & 3 & 8 & 0 & 0 & 61.54%\
maggie & 2 & 2 & 0 & 8 & 0 & 66.67%\
marge & 3 & 1 & 2 & 0 & 4 & 40.00%\
**** & **** & **** & **** & **** & **media** & **62.78%**\
\
\
& bart & homer & lisa & maggie & marge & Acerto:\
bart & 27 & 5 & 2 & 1 & 0 & 77.14%\
homer & 9 & 12 & 3 & 1 & 0 & 48.00%\
lisa & 11 & 1 & 1 & 0 & 0 & 7.69%\
maggie & 4 & 1 & 0 & 7 & 0 & 58.33%\
marge & 6 & 1 & 0 & 0 & 3 & 30.00%\
**** & **** & **** & **** & **** & **media** & **44.23%**\
\
\
& bart & homer & lisa & maggie & marge & Acerto:\
bart & 28 & 4 & 2 & 1 & 0 & 80.00%\
homer & 9 & 11 & 3 & 2 & 0 & 44.00%\
lisa & 11 & 1 & 1 & 0 & 0 & 7.69%\
maggie & 4 & 1 & 0 & 7 & 0 & 58.33%\
marge & 6 & 1 & 0 & 0 & 3 & 30.00%\
**** & **** & **** & **** & **** & **media** & **44.01%**\

Visando melhorar o desempenho da fusão dos classificadores foi realizado
um experimento sem o método de extração de características mais fraco
(em termos de desempenho médio): Histograma da orientação do gradiente.
Foi observado que o desempenho geral e médio melhorou em todos os casos
(exceto o desempenho geral da soma que manteve o mesmo valor). Ainda, em
relação ao desempenho geral, o método Máximo ultrapassou o da Soma, e
alcançou o valor do classificador LDA com Histograma de cor. Sendo, o
método Máximo conseguiu o melhor valor de desempenho médio até agora
($62.78\%$).

Conclusões
==========

Neste trabalho foram comparados três métodos de extração de
características e três métodos de classificação para a base de dados
SIMPSONS. Também foram avaliados cinco métodos de fusão de
classificadores e a influência dos classificadores utilizados na
qualidade da fusão. A qualidade dos métodos de extração de
características na base de imagens foi crucial para o desempenho dos
classificadores, sendo que o método de Histograma de cor foi o método
com melhor desempenho. Entre os métodos de classificação (KNN, LDA e
SVM) o desempenho pode ser considerado equivalente ou próximo. A fusão
dos classificadores não conseguiu sobrepor o baixo desempenho dos
classificadores fracos, que resultaram em um desempenho inferior dos
métodos de fusão do que o LDA com histograma de cor aplicado
individualmente. Buscando suprir esta dificuldade foi realizada uma
análise sem o método de extração de característica mais fraco. O
resultado geral da fusão foi equivalente ao melhor individualmente porém
o desempenho médio foi $1.20\%$ melhor.

O sistema implementado apresenta como pontos fracos os métodos de
extração de características, que não discriminam as classes do problema,
sendo que poderiam ter sido utilizados métodos de extração de
características melhor discriminantes. Além disso a base de imagens
utilizada apresenta poucas imagens de algumas classes, o que dificulta o
aprendizado. Para isso, poderia ter sido aplicadas técnicas pra aumentar
a quantidade de exemplos na base, aplicando alterações nas imagens
existentes. Os parâmetros dos métodos poderiam ter sido melhor
configurados, utilizando técnicas automáticas de configuração de
parâmetros, por exemplo <span>*Iterated racing*</span>. Ou ainda
algoritmos evolutivos multi-objetivo, considerando como objetivos
conflitantes a taxa de classificação em cada classe. Como pontos fortes
do sistema é possível citar a taxa de acerto com histograma de cor. É
possível citar ainda a forma com que foram avaliados os classificadores,
considerando o desempenho médio por classe, o que ameniza o desempenho
de classificadores tendenciosos. É sugerido ainda a comparação com Redes
Neurais Convolucionais que pode apresentar um bom desempenho na
classificação de imagens.

[^1]: <http://www.inf.ufpr.br/lesoliveira/padroes/simpsons.zip>
