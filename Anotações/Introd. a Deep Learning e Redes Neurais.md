# Introdução às Redes Neurais e *Deep Learning*

_**Machine Learning**_: **Treinamento de modelos computacionais a partir de um conjunto de dados para simular a inteligência humana**. Os algoritmos de _machine learning_ aprendem a realizar tarefas específicas com base em exemplos de dados, sem serem explicitamente programados para cada etapa. Eles são amplamente usados em tarefas como classificação, regressão e agrupamento.

_**Deep Learning**_: A aprendizagem profunda (_deep learning_) é uma **subárea de _machine learning_** que **utiliza redes neurais artificiais**, inspiradas no funcionamento do cérebro humano, para realizar tarefas de aprendizado. **A principal característica do _deep learning_ é o uso de redes neurais profundas**, ou seja, **redes com várias camadas ocultas**. Quanto mais camadas uma rede possui, mais capaz ela é de aprender representações abstratas e complexas dos dados. Esse tipo de abordagem é especialmente eficaz em problemas como reconhecimento de imagens, processamento de linguagem natural (PNL) e sistemas de recomendação. 

Para contextualizar, um exemplo real de aplicação de _deep learning_ seria o reconhecimento facial, onde uma rede neural profunda aprende a identificar características faciais, como formato de olhos, nariz e boca, em várias camadas, sem necessidade de intervenção humana para definir esses aspectos.

![Imagem 1](https://github.com/gabrafo/Intro-DL-RN/blob/main/Anexo/Imagem%201.png)

## O que é *Machine Learning*?
Em tarefas onde é difícil de se delimitar um grupo de regras específico para sua conclusão, utiliza-se o conceito de *machine learning*. Em vez de definir regras manualmente, como seria feito em algoritmos tradicionais, **os modelos de _machine learning_ são treinados com dados e aprendem padrões a partir desses exemplos**. Isso permite que os algoritmos façam previsões, classifiquem informações ou tomem decisões sem serem explicitamente programados para cada situação.

Por exemplo, imagine que você queira desenvolver um sistema que identifique e categorize e-mails como "spam" ou "não spam". Seria complicado criar um conjunto de regras que cubra todas as situações possíveis, já que cada e-mail pode ser escrito de forma diferente. Em vez disso, com _machine learning_, você pode treinar um modelo com milhares de exemplos de e-mails rotulados (como "spam" ou "não spam"), e o modelo aprenderá automaticamente as características que diferenciam esses e-mails.

Assim, **usamos dados para que o modelo/algoritmo aprenda padrões a partir de exemplos, permitindo que ele generalize esse conhecimento para novos dados**. Ou seja, uma vez que o modelo foi treinado adequadamente, ele poderá identificar e-mails como "spam" ou "não spam" em novos casos que nunca viu antes, com base nos padrões aprendidos durante o treinamento.

O sucesso de um modelo de _machine learning_ depende muito da qualidade e da quantidade dos dados usados no treinamento, além de sua capacidade de **generalizar**, ou seja, aplicar o conhecimento aprendido a novos dados que ainda não foram apresentados ao modelo.

### Técnicas de aprendizagem
- **Aprendizagem supervisionada**: O modelo é treinado com dados rotulados, ou seja, os exemplos fornecidos já têm suas respectivas categorias ou valores corretos. O objetivo é prever as saídas corretas para novos dados com base no aprendizado dos exemplos anteriores.
    - Exemplo: Um modelo que recebe imagens de gatos e cachorros rotuladas e aprende a distinguir entre os dois.

- **Aprendizagem não supervisionada**: O modelo recebe dados **não rotulados** e precisa identificar padrões ou estruturas nos dados por conta própria. Ele tenta agrupar ou organizar os dados com base em similaridades.
    - Exemplo: Segmentação de clientes em grupos com base em comportamentos de compra.

- **Aprendizagem por reforço**: O modelo aprende através de **tentativa e erro**, recebendo **recompensas** ou **punições** com base em suas ações. O objetivo é maximizar a recompensa ao longo do tempo, encontrando a melhor estratégia possível.
    - Exemplo: Um rato que aprende a navegar por um labirinto, recebendo comida quando se move na direção correta e um choque leve quando se move na direção errada.

Alguns métodos clássicos de machine learning:
- Support Vector Machine (SVM)
- Naive Bayes
- Árvore de Decisão
- Knn
- K-Means

## O que são Redes Neurais?
**Redes neurais** são uma imitação (ainda que simplificada) do neurônio biológico humano, que recebe informações/estímulos e os processa, aplicando os devidos pesos para gerar uma saída. Uma rede neural é composta por **camadas de neurônios artificiais** conectados entre si. Cada camada realiza um processamento dos dados e passa o resultado para a próxima camada, até gerar a saída final.

- **Camada de entrada**: Recebe os dados de entrada.
- **Camadas ocultas**: Processam os dados, ajustando os pesos e realizando cálculos.
- **Camada de saída**: Gera o resultado final do processamento.

As redes neurais existem desde 1950, no entanto, se popularizaram apenas a partir dos anos 2000 devido a alguns fatores, como o maior volume de dados disponível (*Big Data*), maior capacidade de processamento por parte do *hardware* e, consequentemente, melhores GPUs (placas de vídeo).

### Entrada/saída de dados
Redes neurais costumam trabalhar com cálculo vetorial. No exemplo abaixo, temos um exemplo de uma matriz contendo valores "x", onde cada linha representa um atributo (ou *feature*) e cada coluna representa um exemplo (ou instância de treinamento/teste).

![Imagem 3](https://github.com/gabrafo/Intro-DL-RN/blob/main/Anexo/Imagem%203.png)

![Imagem 4](https://github.com/gabrafo/Intro-DL-RN/blob/main/Anexo/Imagem%204.png)

(Imagens do slide do prof. Denilson, disponíveis em suas [videoaulas](https://www.youtube.com/watch?v=2eNwcqHUP8Y&list=PLpAVc-5L0TX_draNYxCmjgm2yYKAy9aIp&index=3&ab_channel=DenilsonAlvesPereira)).

## O que é *Deep Learning*?
É o uso de redes neurais artificiais profundas (com diversas camadas) para gerar modelos matemáticos complexos. Quanto mais camadas, mais complexo é o processamento dos modelos.

Exemplo de rede neural:

![Imagem 2](https://github.com/gabrafo/Intro-DL-RN/blob/main/Anexo/Imagem%202.png)

(Imagem do slide do prof. Denilson, disponível em suas [videoaulas](https://www.youtube.com/watch?v=Au9k8Qyag-g&list=PLpAVc-5L0TX_draNYxCmjgm2yYKAy9aIp&index=3&ab_channel=DenilsonAlvesPereira)).
Para mais arquiteturas de redes neurais, visite: [The mostly complete chart of Neural Networks, explained | by Andrew Tch | Towards Data Science](https://towardsdatascience.com/the-mostly-complete-chart-of-neural-networks-explained-3fb6f2367464).

## Modelo *Perceptron*
### O que é o *Perceptron*?
O ***Perceptron*** é o modelo mais simples de uma **rede neural**. Ele foi desenvolvido por Frank Rosenblatt na década de 1950 e serve como um bloco básico para redes neurais mais complexas. O *Perceptron* realiza uma tarefa de **classificação binária**, ou seja, ele tenta classificar uma (ou mais) entrada(s) em uma de duas categorias (como sim/não, verdadeiro/falso, etc.).
É o modelo mais simples de rede neural.

![Imagem 5](https://github.com/gabrafo/Intro-DL-RN/blob/main/Anexo/Imagem%205.png)

Na imagem acima, consideramos `x1`, `x2` e `x3` como atributos. (Imagem do slide do prof. Denilson, disponível em suas [videoaulas](https://www.youtube.com/watch?v=2eNwcqHUP8Y&list=PLpAVc-5L0TX_draNYxCmjgm2yYKAy9aIp&index=3&ab_channel=DenilsonAlvesPereira)).

#### Estrutura do modelo *Perceptron*
1. Entrada (*input*): 
Imagine que temos várias entradas, cada uma com um valor. Essas entradas são as informações que queremos classificar. Por exemplo, ao tentar predizer clientes perdidos ou ativos em uma determinada companhia de crédito, precisamos informar ao modelo algumas informações sobre esses clientes.
   
Essas informações (características dos clientes) são representadas como `x1`, `x2`, ..., até chegarmos a `xn`, onde:
- x1​ é o valor da primeira característica,
- x2​ é o valor da segunda característica,
- e assim por diante até a enésima característica.

2. Pesos (*weights*): 
Cada entrada tem um **peso** associado a ela, que indica a importância dessa característica na classificação. Esses pesos são representados como `w1`, `w2`, ..., até `wn`. Inicialmente, os pesos são atribuídos aleatoriamente, mas serão ajustados conforme o modelo aprende.

O peso funciona assim:
- Se uma entrada for mais importante, o seu peso será maior.
- Se uma entrada for menos importante, o peso será menor.

3. Cálculo da soma ponderada:
Para processarmos o modelo, precisamos multiplicar cada entrada pelo seu peso correspondente e, em seguida, somar todos esses valores.

**Fórmula do somatório**: 

![Fórmula do somatório](https://latex.codecogs.com/png.image?\inline&space;\large&space;\dpi{150}\bg{black}$$\sum_j&space;x_j\cdot&space;w_j\quad=\quad(x_1\cdot&space;w_1)&plus;(x_2\cdot&space;w_2)&plus;\dots&plus;(x_n\cdot&space;w_n)$$)

- Se a soma ponderada das entradas (multiplicação das entradas pelos seus pesos) for menor ou igual a um *threshold* (viés/limiar), a saída será 0.
- Se a soma ponderada for maior que o *threshold*, a saída será 1.

![Resultado da saída](https://latex.codecogs.com/png.image?%5Cinline%20%5Clarge%20%5Cdpi%7B150%7D%5Cbg%7Bblack%7D%5Ctext%7Bsaida%7D=%5Cbegin%7Bcases%7D0,&%5Ctext%7Bse%7D%5Cquad%5Cmathbf%7Bw%7D%5ET%5Cmathbf%7Bx%7D&plus;b%5Cleq%200%5C%5C1,&%5Ctext%7Bse%7D%5Cquad%5Cmathbf%7Bw%7D%5ET%5Cmathbf%7Bx%7D&plus;b%3E0%5Cend%7Bcases%7D)

**OBS:** O *threshold*/limiar é um valor que **define o ponto de decisão**. Se a soma ponderada for menor ou igual a esse valor, a saída será 0 (o *Perceptron* não "dispara"). Se a soma ponderada for maior, a saída será 1 (o *Perceptron* "dispara").

Se quiséssemos descrever a fórmula de maneira **vetorial** (ou simplificada), teríamos:
- `w` é um **vetor de pesos**.

  ![Imagem 9](https://github.com/gabrafo/Intro-DL-RN/blob/main/Anexo/Imagem%209.png)
  
- `x` é um **vetor de entradas**.

  ![Imagem 8](https://github.com/gabrafo/Intro-DL-RN/blob/main/Anexo/Imagem%208.png)
  
Assim, temos, a seguir, uma representação do **produto escalar** ou **interno** dos vetores `w`  e `x`.

![Representação produto escalar](https://latex.codecogs.com/png.image?\inline&space;\large&space;\dpi{150}\bg{black}$$\mathbf{w}^T\mathbf{x}=(x_1\cdot&space;w_1)&plus;(x_2\cdot&space;w_2)&plus;\dots&plus;(x_n\cdot&space;w_n)$$)

Usando `b` como viés (*bias*), temos que seu valor é o limiar (*threshold*) vezes menos um. Portanto, se o produto escalar mais o valor do viés for maior que zero, teremos um saída um, caso contrário, teremos uma saída zero.

![Imagem 10](https://github.com/gabrafo/Intro-DL-RN/blob/main/Anexo/Imagem%2010.png)

**Exemplo do modelo *Perceptron***:
No exemplo abaixo, temos como atributo de maior peso o estado do tenista (ou seja: se está ou não de bom humor) e, após o cálculo da somatória, percebemos que a decisão de jogar ou não tênis tem uma resposta negativa, visto que o cálculo não ultrapassou o limiar estabelecido de valor 5.

![Imagem 6](https://github.com/gabrafo/Intro-DL-RN/blob/main/Anexo/Imagem%206.png)

(Imagem do slide do prof. Denilson, disponível em suas [videoaulas](https://www.youtube.com/watch?v=2eNwcqHUP8Y&list=PLpAVc-5L0TX_draNYxCmjgm2yYKAy9aIp&index=3&ab_channel=DenilsonAlvesPereira)).

Se valorizássemos mais outro atributo, como a condição climática, teríamos um resultado diferente, com o tenista indo jogar tênis no dia de hoje, já que, nesse caso, a somatória ultrapassaria o limiar de valor 5.

![Imagem 7](https://github.com/gabrafo/Intro-DL-RN/blob/main/Anexo/Imagem%207.png)

(Imagem do slide do prof. Denilson, disponível em suas [videoaulas](https://www.youtube.com/watch?v=2eNwcqHUP8Y&list=PLpAVc-5L0TX_draNYxCmjgm2yYKAy9aIp&index=3&ab_channel=DenilsonAlvesPereira)).

### *Perceptron* Multicamada
Uma rede de *perceptrons* de várias camadas é chamada de *Multilayer Perceptron* (MLP). Diferente de um perceptron simples, que **só pode resolver problemas linearmente separáveis**, o MLP é **capaz de resolver problemas mais complexos e não-linearmente separáveis**, graças à presença de **múltiplas camadas** e à utilização de **funções de ativação não lineares**.

Para ser considerado um *Perceptron* Multicamada, é necessário, ao menos, duas camadas além da camada de entrada (geralmente: uma camada de entrada, uma camada oculta e uma camada de saída).

Essa rede de neurônios é completamente conectada, todos os neurônios "conversam" uns com os outros, então cada saída de um *perceptron* é "replicada" como entrada para todos os neurônios subsequentes.

#### O que é um problema "linearmente separável" e como esse tipo de problema se relaciona com *perceptrons*?
Imagine que você está organizando dois tipos de objetos em uma mesa: **maçãs** e **laranjas**. Se você conseguir **desenhar uma linha reta** no meio da mesa, de modo que todas as maçãs fiquem de um lado da linha e todas as laranjas fiquem do outro lado, isso significa que esses dois grupos de objetos são **linearmente separáveis**.

Representando as maçãs como `o` e as laranjas como `x`, teríamos:
```
Exemplo linearmente separável:

o o o o o  <-- Maçãs
-----------
x x x x x  <-- Laranjas

Exemplo não linearmente separável:

x o x o x 
-----------
o x o x o  

De ambos os lados da fronteira imaginária existem elementos tanto da
classe laranja, como da classe maçã.
```

**Linearmente separável** significa simplesmente que é possível separar dois grupos de coisas com uma linha reta (ou, em casos mais complexos, com uma divisão "reta", como um plano ou hiperplano).

Um ***Perceptron* Simples** faz uma coisa bem específica: ele tenta encontrar uma **reta** (em 2D) ou um **hiperplano** (em mais dimensões) que separe os dados em duas classes.

Vamos ver o processo básico:
1. O ***perceptron*** recebe dados de entrada (que podem ter 2, 3 ou mais características).
2. Ele combina essas entradas de uma forma linear (soma ponderada) e, com base nisso, toma uma decisão.
3. A "decisão" do *perceptron* pode ser vista como traçar uma linha reta (ou um hiperplano) para separar as classes.

Se os dados forem linearmente separáveis (como no exemplo das maçãs e laranjas organizadas perfeitamente), o *perceptron* consegue encontrar essa reta. Mas se os dados **não forem linearmente separáveis**, com as frutas misturadas, por exemplo, ele não conseguirá encontrar uma solução.

![Imagem 14](https://github.com/gabrafo/Intro-DL-RN/blob/main/Anexo/Imagem%2014.png)

Já um ***Perceptron* Multicamada** adiciona mais camadas de neurônios, permitindo que a rede combine informações de formas mais complexas. Em vez de apenas desenhar uma linha reta para separar as classes, ele pode **aprender curvas e fronteiras não lineares** para fazer uma separação mais sofisticada.

Um exemplo de problema que o *Perceptron* Simples não consegue resolver, mas um MLP consegue é a operação XOR (*Exclusive Or*). 

A operação XOR retorna 1 **somente quando** uma das entradas é 1 e a outra é 0 (ou seja, quando as entradas são diferentes).

![Imagem 11](https://github.com/gabrafo/Intro-DL-RN/blob/main/Anexo/Imagem%2011.png)

Repare que, mesmo que tentemos, não vamos conseguir traçar uma linha que divida as classes entre "0" e "1" de maneira exata. O *Perceptron* Simples só pode traçar uma **reta** (ou plano) para separar os dados, e isso não é suficiente para resolver o XOR, que exige uma separação **não linear**, que o MLP consegue resolver.

Exemplo de cálculo de um *Perceptron* Multicamada sem considerarmos a função de ativação:

![Imagem 12](https://github.com/gabrafo/Intro-DL-RN/blob/main/Anexo/Imagem%2012.png)

(Imagem do slide do prof. Denilson, disponível em suas [videoaulas](https://www.youtube.com/watch?v=5I5nYybVjZ0&t=224s&ab_channel=DenilsonAlvesPereira)).

### Função de ativação
Cada neurônio é caracterizado pelo peso, *bias* (viés) e a função de ativação. Enquanto os neurônios fazem uma transformação linear na entrada pelos pesos e *bias*, a função de ativação faz uma transformação **não linear**. Isso é o que torna o MLP tão diferente do SLP (*Single Layer Perceptron*).

![Imagem 13](https://github.com/gabrafo/Intro-DL-RN/blob/main/Anexo/Imagem%2013.png)

(Imagem do slide do prof. Denilson, disponível em suas [videoaulas](https://www.youtube.com/watch?v=5I5nYybVjZ0&t=224s&ab_channel=DenilsonAlvesPereira)).

#### Função Sigmóide
É um exemplo de função de ativação, mapeando a saída `z` para o intervalo (0,1) e é usada frequentemente em redes neurais para introduzir não linearidade. Além disso, aparece, principalmente, na camada de saída para problemas de **classificação binária** e não é muito eficaz quando os gradientes se tornam muito pequenos.

![Imagem 15](https://github.com/gabrafo/Intro-DL-RN/blob/main/Anexo/Imagem%2015.png)

![Imagem 16](https://github.com/gabrafo/Intro-DL-RN/blob/main/Anexo/Imagem%2016.png)

(Imagens do slide do prof. Denilson, disponíveis em suas [videoaulas](https://www.youtube.com/watch?v=5I5nYybVjZ0&t=224s&ab_channel=DenilsonAlvesPereira)).

#### Função TanH (tangente hiperbólica)
A função tangente hiperbólica (tanh⁡(x)) é uma derivação da função sigmoide. No entanto, ela é **simétrica em relação à origem** (o ponto tanh⁡(0) está no centro do gráfico). A função tanh⁡(x) tem um **intervalo de saída entre -1 e 1**, enquanto a sigmoide tem saída entre 0 e 1. Isso permite que a **tanh⁡(x) lide melhor com valores negativos e positivos em entradas de redes neurais**, ajudando a centrar os dados em torno de zero, o que pode melhorar o desempenho e a convergência durante o treinamento.

![Imagem 17](https://github.com/gabrafo/Intro-DL-RN/blob/main/Anexo/Imagem%2017.png)

(Imagens do slide do prof. Denilson, disponíveis em suas [videoaulas](https://www.youtube.com/watch?v=5I5nYybVjZ0&t=224s&ab_channel=DenilsonAlvesPereira)).

#### Função ReLU (*Rectified Linear Unit*)
Essa é uma das funções de ativação mais populares usadas em redes neurais profundas, especialmente em redes convolucionais (CNNs). Ela retorna o valor da entrada x se x for positivo; caso contrário, retorna 0. Graficamente, a ReLU é uma linha reta com inclinação 1 para valores positivos e e colada ao eixo x para valores negativos.
Além disso, a função ReLU não ativa todos os neurônios ao mesmo tempo, o que torna a rede esparsa e a computação se torna fácil e eficiente.
Uma de suas desvantagens é que apresenta problemas quando os gradientes são muito pequenos (próximos de zero).

![Imagem 16](https://github.com/gabrafo/Intro-DL-RN/blob/main/Anexo/Imagem%2018.png)

(Imagens do slide do prof. Denilson, disponíveis em suas [videoaulas](https://www.youtube.com/watch?v=5I5nYybVjZ0&t=224s&ab_channel=DenilsonAlvesPereira)).

##### Problema do neurônio morto
Um **neurônio morto** em redes neurais ocorre quando uma unidade de ativação (ou neurônio) deixa de atualizar seu valor de saída durante o treinamento, ou seja, ela **sempre retorna zero** independentemente da entrada. Esse problema é comumente associado à função de ativação **ReLU**.

Na função **ReLU** (ReLU(x) = max⁡(0,x)), qualquer valor de entrada negativo resulta em uma saída de 0. Se muitos neurônios recebem entradas negativas repetidamente, esses neurônios podem **parar de aprender** porque o gradiente para esses valores negativos também será zero. Quando isso acontece, esses neurônios se tornam "mortos" ou inativos, pois suas saídas continuam sendo 0 durante todo o treinamento, sem a capacidade de mudar.

A ***Leaky* ReLU** (que veremos a seguir) resolve parcialmente o problema do neurônio morto ao permitir que valores negativos passem com um pequeno coeficiente α. Isso garante que, mesmo quando as entradas forem negativas, a ativação não será completamente zero, mas sim um valor pequeno. Com isso, o neurônio ainda tem a chance de continuar aprendendo e ajustando seus pesos.

#### Função *Leaky* ReLU (*Leaky Rectified Linear Unit*)
É uma versão da ReLU, que busca diminuir problemas como o do "neurônio morto", citado acima. A grande diferença entre as duas é que, para valores negativos, utiliza-se uma constante α (que costuma ter um valor muito pequeno) para auxiliar no cálculo.

![Leaky ReLU](https://latex.codecogs.com/png.image?\dpi{110}\bg{black}$$\;\text{Leaky&space;ReLU}(x)=\begin{cases}x&\text{se}x\geq&space;0,\\\alpha&space;x&\text{se}x<0\end{cases}\;$$)

![Imagem 19](https://github.com/gabrafo/Intro-DL-RN/blob/main/Anexo/Imagem%2019.png)

#### Função *Softmax*
É um tipo de função sigmóide usada para converter uma lista de números reais (que podem ser positivos, negativos ou zero) em probabilidades. Cada número é transformado em um valor entre 0 e 1, e todos os valores juntos somam 1. Isso é especialmente útil em problemas de classificação multiclasse, onde queremos saber a probabilidade de uma entrada pertencer a diferentes classes. Além disso, ela não é usada em cada camada oculta da rede neural, mas apenas na camada de saída.

![Imagem 20](https://github.com/gabrafo/Intro-DL-RN/blob/main/Anexo/Imagem%2020.png)

## Função de custo
A **função de custo** é uma métrica que avalia o quão bem o modelo está realizando suas previsões. Ela calcula a diferença entre a saída prevista pelo modelo (`ŷ`) e o valor real esperado (`y`), fornecido pelos dados de treinamento. 

Quando treinamos uma rede neural, o objetivo é ajustar os **pesos** (`w`) e os ***bias***(`b`) de tal forma que a saída prevista (`ŷ(i)`) para uma dada entrada (`x(i)`) se aproxime o máximo possível da saída correta (`y(i)`). Quanto menor essa diferença, melhor o modelo se torna em suas previsões.

### Função de Custo Quadrático (MSE - *Mean Squared Error*)

