# churn_predict

## Descrição do dataset
A base de dados utilizada na construção do modelo é composta por 10.000 registros, e 14 variáveis, sendo elas:
- RowNumber: O número da linha do registro
- CustomerId: O ID único do cliente
- Surname: O primeiro nome do cliente
- CreditScore: A pontuação de crédito
- Geography: O país de residência do cliente
- Gender: O gênero do cliente
- Age: A idade do cliente
- Tenure: O tempo de vínculo com a empresa
- Balance: O saldo do cliente no momento do registro
- NumOfProducts: Número de produtos/serviços que o cliente possui
- HasCrCard: Indica se o cliente possui ou não um cartão de crédito (variável binária)
- IsActiveMember: Indica se o cliente é ativo no banco (critério de cliente ativo não considerado) (variável binária)
- EstimatedSalary: Salário estimado do cliente
- Exited: Indica se o cliente deixou a empresa (churn) - Variável Target

Descrição das variáveis retiradas da seguinte fonte encontrada:
https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn

## Análise exploratória dos dados
A Análise exploratória dos dados foi realizada com todas as variáveis individualmente, utilizando alguns métodos estatísticos, como:
- Estatística descritiva geral
- Análise de distribuição
- Análise de outliers
- Análise de correlação
- Para as variáveis contínuas com grande intervalo de valores entre si, foi utilizado um método de agrupamento por faixas (qcut), onde ele corta os valores por faixas, para ter maior precisão nas análises.


foi observado então que:
Entre os países, a Alemanha é o país que mais tem churn positivo proporcional (32%) de acordo com a população de cada país na base de dados.

Na distribuição e análise de idade foram encontrados outliers e valores que não podem ser levados em consideração para as análises estatística. Porém, estes valores não serão removidos para garantir a qualidade do modelo para os dados futuros.

Em geral, a faixa de score de 300 a 499 é a que possui proporcionalmente a maior taxa de churn, com 24%. Mas ainda assim, essa variável se mostrou não ter relevância o suficiente na decisão de churn.

Foi notado mais de 30% dos registros possuem Balance = 0. Desses, somente 14% possuem Churn. 

Analisando por faixas, vimos que de modo geral, o churn de quem tem balance = 0 é o mais alto, mas de maneira equilibrada referente as outras faixas de balance.

Foi constatado o valor "4" da variável NumOfProducts como Outlier, tendo em vista que só há 0.6% dos registros para este valor. Ainda assim não será removido pois na base de teste (desconhecido) poderá ocorrer este valor

foi constatado que 100% do valor 4 possui churn positivo, o que pode influenciar o modelo a sempre pesar para o churn positivo quando cair este valor nos dados novos. Essa variável será observada na escolha no modelo.

A proporção da base de quem tem cartão de crédito é de 70%, porém de cada um dos valores, a proporção de churn são semelhantes entre eles.

Para a variável de Membros ativos, a porcentagem de churn para cada uma das classes também está equilibrada, o que irá ajudar o modelo com os dados novos


A faixa salarial também possui equilibrio entre todas as faixas quando se fala sobre Churn positivo. A distribuição é normal e não há outliers.


Foi constatado que não há nenhuma variável que possua forte correlação com a variável Exited, tanto negativa quanto positiva.


### Tratamento

- Foram removidas algumas variáveis da base de dados como (RowNumber, Surname e CustomerID) pois não são relevantes para a análise e construção do modelo.
- A variável Gender também foi removida, para não haver discriminação e generalização do modelo, podendo resultar em enviezamento.

- Todos os dados continham todos os valores preenchidos com os formatos corretos, não havendo necessidade de transformar ou preencher com algum método específico.
- Os outliers observados não foram removidos, pois não foram identificados como erros de população para as variáveis. Outliers foram mantidos para o modelo aprender com possíveis outliers também nos dados futuros desconhecidos.

### Pré Processamento
As variáveis da base de dados foram separadas em:
- Categóricas
- Contínuas
- Binárias

Nara as variáveis Categóricas, utilizei o método OneHotEncoder, para transformar valores do tipo objeto ou string em classes (França : 1, Alemanha: 2)
Nas variáveis contínuas utilizei o método StardardScaler para normalizar os valores e não correr o risco de enviezamento com peso dos valores originais
As variáveis binárias não necessitaram de nenhum pré processamento, visto que já estavam no padrão ideal (0 e 1)

Para realizar o pré-processamento com esses métodos citados, utilizei o Pipeline, que organiza e ordena esses processamentos, e já aplica ao modelo de forma automatizada, em linhagem. Esse mesmo Pipeline automatizará o processo de pré processamento para os novos dados.

## Escolha do modelo

Como a maioria das variáveis do dataset não possuia distribuição normal, optei por escolher algoritmos de árvore, que tem mais facilidade em interpretar dados não-lineares e tem maior resistência a outliers.


Para decidir qual modelo seguir, utilizei o método do gridsearch, que faz uma otimização de modelos e hiperparâmetros para encontrar as melhores combinações possíveis. Os modelos listados foram: RandomForest, LGBM, GradientBoosting.

Como resultado o gridsearch trouxe o LGBM.

### Métricas de avaliação

Por mais que gridsearch modelo tenha recomendado o LGBM, a decisão final foi por seguir com o RandomForest, pois a principal métrica de avaliação que utilizei foi o KS (Kolmogorov-Smirnov), que mostra a capacidade de distinguir as duas classes. 
Esta métrica é muito utilizada e recomendada principalmente para classificação binária, com modelo de negócio financeiro ou churn.

Avaliando esta métrica, o modelo de RandomForest teve uma maior distância máxima em distinção das classificações (56% de distância), além de uma curva ROC-AUC de 89%.



## Conclusões gerais




