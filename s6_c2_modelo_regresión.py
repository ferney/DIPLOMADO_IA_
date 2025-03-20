# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import drive
drive.mount('/content/drive')

import os # leer patch ruta sde ecarpetas
# chdir: cambia la dirección
os.chdir('/DATA')

os.listdir() # Lista de los documentos

df = pd.read_excel('marketing_extract_2X_Income (1).xlsx')

"""ANÁLISIS DESCRIPTIVO"""

df.info()

df.shape

df.describe()

"""ANÁLISIS EXPLORATORIA"""

x = df['MntMeatProducts']
y = df['Income']
plt.scatter(x,y)
plt.title('Ingreso anual respecto a las compras de carne')
plt.xlabel('MntMeatProducts: Compras de carne')
plt.ylabel('Income: Ingreso anual')

"""ELIMINAR DATOS ATIPICOS VARIABLE X y y

> Añadir blockquote


"""

plt.subplot(1,2,1)
plt.boxplot(x)
plt.title('x: Compras de carne')
plt.subplot(1,2,2)
plt.boxplot(y)
plt.title('y: Ingreso anual')
plt.show()

x_Q1 = x.quantile(0.25)
x_Q3 = x.quantile(0.75)
x_IQR = x_Q3 - x_Q1

x_IQR

y_Q1 = y.quantile(0.25)
y_Q3 = y.quantile(0.75)
y_IQR = y_Q3 - y_Q1



df = df[(df['MntMeatProducts'] >= x_Q1 - 1.5*x_IQR) & (df['MntMeatProducts'] <= x_Q3 + 1.5*x_IQR)]
df = df[(df['Income'] >= y_Q1 - 1.5*y_IQR) & ( df['Income']<= y_Q3 + 1.5*y_IQR)]

X.shape

Y.shape

"""# DATOS LIMPIOS O PROCEDADOS"""

x = df['MntMeatProducts']
y = df['Income']
plt.scatter(x,y)
plt.title('Ingreso anual respecto a las compras de carne')
plt.xlabel('MntMeatProducts: Compras de carne')
plt.ylabel('Income: Ingreso anual')

"""MODELO DE MARCHI LERNING

# Entrenar el modelo de machine learning
"""

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# Paso 1 Definir el conjunto de entrenamiento
# Target : Objetivo es la variable  a predecir
# y: Income
y = df['Income']

# Carcaterísticas: atributos
x = df[['MntMeatProducts']]

# Paso 2. Hacer la división de los conjuntos para entrenar (fit) y probar(test)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.2) # 80 train 20 test

# Paso 3. Definir el modelo
modelo = LinearRegression()

# Paso 4. Entrenar: es el proceso de ajuste para obtener el modelo
modelo.fit(x_train ,y_train)

modelo.coef_

"""GRAFICAR"""

X_range = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)  # Valores equidistantes
y_range = modelo.predict(X_range)  # Predicción para la línea recta

x = df['MntMeatProducts']
y = df['Income']
plt.scatter(x,y)
plt.title('Ingreso anual respecto a las compras de carne')
plt.xlabel('MntMeatProducts: Compras de carne')
plt.ylabel('Income: Ingreso anual')

plt.plot(X_range, y_range, color='red', label='Línea de regresión')
plt.legend()
# x cuento compra en carne y: cuanto ingreso en dinero

X_prueba = 100
y_predicha = modelo.predict([[X_prueba]])
plt.scatter(X_prueba, y_predicha,  color='green', label='Punto de prueba')
plt.legend()

