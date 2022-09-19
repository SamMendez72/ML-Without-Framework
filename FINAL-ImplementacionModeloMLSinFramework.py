'''
Samuel Méndez Villegas - A01652277

En este archivo se presenta la implementación de un modelo de machine learning, siendo el seleccionado un árbol de decisión. 
El código es implementado desde 0, y está inspirado en el código desarrollado por Ander Fernández Jauregui. 

La base de datos que se utiliza, se llama `500_Person_Gender_Height_Weight_Index.csv` y registros de personas como su peso, estatura, género
 y si cuentan con obesidad. Por lo tanto el objetivo del modelo será predecir con 3 variables independientes si una persona tiene obesidad.
 
'''

## Estas son las únicas librerías que utilizaremos, las cuales nos permiten realizar operaciones con arrays y 
## y en data frames.
import pandas as pd
import numpy as np
import itertools # Se utiliza para realizar algunas operaciones de combinaciones

df = pd.read_csv('500_Person_Gender_Height_Weight_Index.csv')
df.head()

df['Obese'] = (df.Index >= 4).astype('int').astype('str')
df.drop('Index', axis = 1, inplace = True)

## Se obtiene la entropia de un conjunto de datos
def entropia(y):
    p = y.value_counts()/y.shape[0] # almacena el valor de p y 1-p
    entropy = np.sum(-p*np.log2(p)) # aplica la operacion para p y 1-p
    return entropy

## Se obtiene la información de ganancia a partir de la entropía de un conjunto de datos
def gananciaInformacion(y, filtro):
    p = sum(filtro)
    q = filtro.shape[0] - p
    ganancia_info = entropia(y) - p/(p+q)*entropia(y[filtro]) - q/(p+q)*entropia(y[-filtro])
    return ganancia_info

# Se obtienen todas las combinaciones posibles de una variable para evaluar la ganancia de cada combinación y
## de esta forma decidir el corte
def combinacionesPosibles(x):
    combinaciones = []
    
    x = x.unique()
    
    for i in range(0, len(x) + 1):
        for j in itertools.combinations(x,i):
            j = list(j)
            combinaciones.append(j)
            
    return combinaciones[1:-1]

## Se calcula el mejor corte del árbol dado una variable predictora, y la variable objetivo
def evaluacionCorte(x, y):
    valores_corte = []
    ganancias = []
    
    ## Evalúa si la variable es numérica
    if (x.dtypes != 'O'):
        combinaciones = x.sort_values().unique()[1:]
        numerica = True
    
    ## Si la variable no es numérica, entonces es categórica
    else:
        combinaciones = combinacionesPosibles(x)
        numerica = False
    
    ## Se calcula la ganancia de información para cada valor de la variable
    for opcion in combinaciones:
        if (numerica == True):
            filtro = x < opcion
        else:
            filtro = x.isin(opcion)
            
        ## Se obtiene la ganancia de información con el filtro correspondiente
        ganancia_opcion = gananciaInformacion(y, filtro)
        ganancias.append(ganancia_opcion)
        valores_corte.append(opcion)
        
      # Check if there are more than 1 results if not, return False
    if len(ganancias) == 0:
        return(None,None,None, False)
        
    ## Se obtiene la ganancia de información más alta
    mejor_ganancia = max(ganancias)
    mejor_ganancia_index = ganancias.index(mejor_ganancia)
    mejor_corte = valores_corte[mejor_ganancia_index]
    
    return (mejor_ganancia, mejor_corte, numerica, True)


## Se recibe el data frame y se obtiene la mejor información de ganancia para cada una de las variables. Por lo tanto regresa
## la variable en donde se hará el corte, el valor del corte y la ganancia del corte.
def mejorCorte(y, df):
    filtros = df.drop(y, axis = 1).apply(evaluacionCorte, y = df[y])
    
    ## Se obtienen solamente los filtros que pueden dividir el set
    filtros = filtros.loc[:,filtros.loc[3,:]]
    
    ## Se obtienen los resultado
    variable_corte = max(filtros)
    valor_corte = filtros[variable_corte][1]
    ganancia_corte = filtros[variable_corte][0]
    corte_numerico = filtros[variable_corte][2]
    
    return (variable_corte, valor_corte, ganancia_corte, corte_numerico)

## Se realiza el corte econtrado en la función anterior. El set de datos se divide en dos, los que cumplen con la condición del corte, y los 
## que no.
def realizarCorte(x, valor, df, es_numerico):
    if es_numerico:
        df_1 = df[df[x] < valor]
        df_2 = df[(df[x] < valor) == False]
    else:
        df_1 = df[df[x].isin(valor)]
        df_2 = df[(df[x].isin(valor)) == False]
    return df_1, df_2

## Se realiza la predicción de cuál será el valor o categoría del registro. 
def hacerPrediccion(df, target_factor):
    if target_factor:
        prediccion = df.value_counts().idxmax()
    else:
        prediccion = df.mean()
    return prediccion

## Entenamiento completo del árbol de decisión. Se incluyen hiperparámetros como la profundidad máxima y el mínimo de ganancia para realizar
## los cortes. 
def entrenamiento(df, y, target_factor, max_depth = None,min_information_gain = 1e-20, counter = 0):

    if max_depth == None:
        depth_cond = True
    else:
        if counter < max_depth:
            depth_cond = True
        else:
            depth_cond = False
        
    if depth_cond:
        var, val, ig, var_type = mejorCorte(y, df)
        
        if ig is not None and ig >= min_information_gain:
            counter += 1
            izquierda, derecha = realizarCorte(var, val, df, var_type)
            
            tipo_corte = '<=' if target_factor else 'in'
            pregunta = "{} {} {}".format(var, tipo_corte, val)
            subtree = {pregunta: []}
            
            si = entrenamiento(izquierda, y, target_factor, max_depth, min_information_gain, counter)
            no = entrenamiento(derecha, y, target_factor, max_depth, min_information_gain, counter)
            
            if si == no:
                subtree = si
            else:
                subtree[pregunta].append(si)
                subtree[pregunta].append(no)
            
        else:
            pred = hacerPrediccion(df[y], target_factor)
            return pred
        
    else:
        pred = hacerPrediccion(df[y], target_factor)
        return pred
    
    return subtree

## Implementación del entrenamiento del modelo:
max_depth = 5
min_information_gain  = 1e-5

decisiones = entrenamiento(df,'Obese', True, max_depth,min_information_gain) # Obese es la variable objetivo

# Se muestra en consola el árbol generado. 
print('El árbol generado a partir de los datos de entrenamiento es el siguiente:')
print(decisiones)

