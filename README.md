# Previsão do Consumo de Água por Unidade

Este projeto tem como objetivo prever o consumo de água de unidades residenciais utilizando regressão linear. O consumo é calculado pela diferença entre as variáveis "final" e "inicial" do conjunto de dados. O processo envolve análise exploratória, tratamento de dados, codificação de variáveis categóricas, construção de modelos de regressão e avaliação estatística dos coeficientes.

## Objetivos
- Prever o consumo de água por unidade.
- Tratar valores nulos e inválidos (ex: consumos negativos).
- Construir um modelo inicial com todas as variáveis, incluindo o bloco.
- Identificar variáveis estatisticamente insignificantes (p-valor > 0.05) e refazer o modelo apenas com as variáveis significativas.

## Etapas do Projeto

### 1. Importação das Bibliotecas
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import statsmodels.api as sm
```

### 2. Leitura do Dataset
```python
df = pd.read_csv('water_consumption - water_consumption.csv')
df.head()
```

### 3. Estatísticas Descritivas
```python
print("Estatísticas descritivas iniciais:")
print("=================================")
df.describe()
```

### 4. Análise da Variável Categórica 'block'
```python
df["block"].value_counts()
```

### 5. One-Hot Encoding da Variável 'block'
Transforma a variável categórica 'block' em variáveis binárias:
```python
df_copy = df.copy()
block_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
for block in block_list:
    df_copy[block] = df_copy['block'].apply(lambda x: 1 if x == block else 0)
df_copy = df_copy.drop('block', axis=1)
```

### 6. Tratamento de Valores Nulos
Remove linhas com valores nulos:
```python
df_copy.dropna(inplace=True)
df_copy.reset_index(drop=True, inplace=True)
```

### 7. Remoção de Leituras Inválidas
Remove leituras onde 'initial' ou 'final' são zero:
```python
df_copy = df_copy[(df_copy["initial"] != 0) & (df_copy["final"] != 0)]
```

### 8. Cálculo do Consumo
Cria a variável alvo 'consume':
```python
df_copy['consume'] = df_copy['final'] - df_copy['initial']
df_copy.describe()
```

### 9. Remoção de Consumos Negativos
```python
df_copy = df_copy[(df_copy["consume"] > 0)]
```

### 10. Modelo Inicial de Regressão Linear
Utiliza todas as variáveis, inclusive os blocos:
```python
model = LinearRegression()
x = df_copy[['price','month','year','apartment','A','B','C','D','E','F','G']]
y = df_copy['consume']
model.fit(x, y)
print(model.score(x, y))
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
print(model.summary())
```

### 11. Seleção de Variáveis Significativas
Retira variáveis com p-valor > 0.05 e refaz o modelo:
```python
model = LinearRegression()
x = df_copy[['price','month','year']]
y = df_copy['consume']
model.fit(x, y)
print(model.score(x, y))
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
print(model.summary())
```

## Observações
- O tratamento de dados é essencial para evitar erros na modelagem.
- O modelo final considera apenas variáveis estatisticamente significativas.
- A análise dos coeficientes é feita pelo p-valor no resumo do modelo OLS.
