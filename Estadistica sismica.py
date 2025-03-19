# -*- coding: utf-8 -*-

import pandas as pd  # Para el manejo de los datos
import numpy as np # Para el manejo de operaciones entre datos, y las columnas
import matplotlib.pyplot as plt # Para el manejo de las gráficas
import seaborn as sns # Para el manejo de gráficas

# Cargar el dataset 
file_url = "./INFO_SISMO_PETROLEO_2D.csv"
df = pd.read_csv(file_url, encoding="latin1")

# Mostrar información general del dataset
df.info()
print("\nPrimeras filas del dataset:")
print(df.head())

# Revisar valores nulos
print("\nValores nulos por columna:")
print(df.isna().sum())

# Estadísticas descriptivas
describe_df = df.describe()
print("\nEstadísticas descriptivas:")
print(describe_df)

# Histograma de la longitud de líneas sísmicas
plt.figure(figsize=(8, 5))
sns.histplot(df["LINE_LENGT"], bins=50, kde=True, color="blue")
plt.title("Distribución de la Longitud de Líneas Sísmicas")
plt.xlabel("Longitud (km)")
plt.ylabel("Frecuencia")
plt.show()

# Boxplot para detectar valores extremos en la longitud de líneas sísmicas
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["LINE_LENGT"], color="orange")
plt.title("Boxplot de Longitud de Líneas Sísmicas")
plt.xlabel("Longitud (km)")
plt.show()

# Conteo de líneas sísmicas por cuenca
plt.figure(figsize=(10, 6))
sns.countplot(y=df["CUENCA"], order=df["CUENCA"].value_counts().index, palette="viridis")
plt.title("Número de Líneas Sísmicas por Cuenca")
plt.xlabel("Cantidad")
plt.ylabel("Cuenca")
plt.show()

# Relación entre longitud de líneas sísmicas y año de adquisición
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df["ANO_ADQ"], y=df["LINE_LENGT"], alpha=0.5, color="red")
plt.title("Longitud de Líneas Sísmicas por Año de Adquisición")
plt.xlabel("Año de Adquisición")
plt.ylabel("Longitud de la Línea (km)")
plt.show()