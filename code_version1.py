
## Implementación de un modelo de ML sin un Frame Work

# Importación de librerías a utilizar
import numpy as np
import pandas as pd

# Funciones a implementar

# Función que calcula la impuridad gini de una variable
def giny_impurity(x):
    p = x.value_counts()/x.shape[0]
    gini = 1-np.sum(p**2)
    return(gini)

# Función que obtiene el peso (con el valor gini) de las hojas
def total_giny_impurity(x, impurities):
    # to do
    
# pruebas TO DO
df['work_year'].value_counts()

p = df['work_year'].value_counts()/df['work_year'].shape
p

df['work_year'].value_counts
giny_impurity(df['work_year'])


# datos a utilizar
df = pd.read_csv('ds_salaries-Copy1.csv')
df.head()

# variables a utilizar
df = df.drop(['Unnamed: 0', 'job_title', 'salary_currency'], axis = 1)
df.head()