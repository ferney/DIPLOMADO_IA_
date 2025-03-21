# -*- coding: utf-8 -*-


## Importar las librerías necesarias
import pandas as pd # manipular datos, dataframe
import matplotlib.pyplot as plt # graficar Seaborn


from sklearn.model_selection import train_test_split
# train: Datos para entrenar
# test: probar
# split: partición de los datos

from sklearn.tree import DecisionTreeClassifier
# .tree: usar árboles de decisión

from sklearn import tree
from sklearn.metrics import classification_report, accuracy_score

'''data = {
    'Age': [16, 17, 20, 25, 35, 40, 45, 50, 22, 28, 30, 33, 38, 42, 46, 52, 18, 26, 55, 60],
    'Weight': [55, 65, 70, 80, 68, 85, 72, 90, 60, 78, 74, 82, 77, 88, 75, 92, 68, 83, 95, 100],
    'Smoker': [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],  # 0 = No fumador, 1 = Fumador
    'Risk': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 0 = Bajo riesgo, 1 = Alto riesgo
}'''
# se carga mas información para dar mas precición heart_attack_risk_dataset.csv
data = {
    "Age": [16, 17, 20, 25, 35, 40, 45, 50, 22, 28, 30, 33, 38, 42, 46, 52, 18, 26, 55, 60, 15, 19, 21, 27, 36, 41, 44, 51, 23, 29, 31, 34, 39, 43, 47, 53, 19, 24, 56, 61, 14, 18, 22, 26, 34, 39, 43, 49, 21, 27, 32, 35, 37, 41, 48, 54, 17, 25, 58, 62],
    "Weight": [55, 65, 70, 80, 68, 85, 72, 90, 60, 78, 74, 82, 77, 88, 75, 92, 68, 83, 95, 100, 54, 66, 71, 81, 69, 86, 73, 91, 61, 79, 75, 83, 78, 89, 76, 93, 67, 84, 96, 101, 53, 67, 72, 82, 70, 87, 74, 92, 62, 80, 76, 84, 79, 90, 77, 94, 66, 85, 97, 102],
    "Smoker": [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    "Risk": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}


# Convertir el dataset a un DataFrame de pandas
df = pd.DataFrame(data)

# Mostrar los datos
print("Datos del dataset:")
print(df)

"""**Q1** ¿A qué decision podríamos llegar con estos datos?
RTA: Gestionar el riesgo de una paciente

**Q2** ¿Que tipo de problema debemos abordar?
RTA: Clasificación

# Actividad 1. Separar las características (X) de la variable objetivo (y)

* Cuál es la variable objetivo? Rta:  Risk
* Cuáles son los parámetros(Columnas, características, entradas)?: Age, Weight, Smoker

# Actividad 2. Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
"""



# y por notación el target (hipotesis/predicción)
y = df['Risk']

# en X se agrupan las caracterísitcas
x = df[['Age', 'Weight', 'Smoker']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)



"""# Actividad 3. Seleccione y ajuste el modelo: árbol de decision

* Ejecute lás líneas de código, si desea cambie el nombre del modelo


"""

# Crear el modelo de árbol de decisión
modeloDTC = DecisionTreeClassifier()

# Entrenar el modelo (fit: ajuste)
modeloDTC.fit(x_train, y_train)

"""# Actividad 4. Hacer predicciones sobre los datos de prueba

* Aplique al modelo la palabra predict y evalue en los datos de prueba y guardelos en y_pred


```
modelo.predict(datos_de_prueba)
```


"""

modeloDTC

x_test

x_train

y_test

# Usar el modelo con el método .predic va

y_pred = modeloDTC.predict(x_test)
y_pred

"""# Actividad 5. Evalue el modelo

* Use la medida de accuracy para medir el desempeño del modelo

```
accuracy_score(y_test, y_pred)
```
* Genere un reporte de medidas del modelo con

```
classification_report(y_test, y_pred)
```


* Imprima las mediciones y acopmpañe de valores que orienten al usuario
"""



print("Reporte de Clasification")
target_names = ['Bajo Riesgo (0)', 'Alto Riesgo (1)']

print(classification_report(y_test, y_pred,  target_names = target_names))
report = classification_report(y_test, y_pred)

"""# Actividad 6. Visualizar el árbol de decisión


"""

plt.figure(figsize=(10, 8))
tree.plot_tree(modeloDTC, feature_names=['Age', 'Weight', 'Smoker'], class_names=['Low Risk', 'High Risk'], filled=True)
plt.title("Árbol de Decisión - Riesgo de Ataque Cardíaco")
plt.show()

"""# Actividad 7. Ejemplo de predicción

* Escriba un código para evaluar un dato del modelo

* Use estos comentarios que orienten al usuario



```
print("\nPredicción con un ejemplo nuevo:")
print(f"Predicción (0 = Bajo Riesgo, 1 = Alto Riesgo): {prediction[0]}")

```

# Parte 2. Cambiar el tamaño del dataset

* Use el archivo >> DataSet/Classification/Health/heart_attack_risk_dataset.csv

* Ejecute nuevamente y analise los resultados
"""



"""#  Con el dataset primario

Precisión General (Accuracy = 0.75): El modelo clasifica correctamente el 75% de las muestras (3 de 4 casos de prueba).

Problema: La muestra de prueba es muy pequeña, por lo que el resultado no es estadísticamente confiable. Un solo error afecta drásticamente la métrica.

Existe un inconveniente por que los datos de "Bajo Riesgo" tiene muy pocas muestras, lo que dificulta su aprendizaje y el modelo podría estar memorizando patrones específicos del conjunto de entrenamiento en lugar de generalizar.

Nos debemos asegurar que los datos están balanceados. Si hay más datos de una clase que de otra, el modelo se puede estar sesgar, adicional que el dataset es pequeño y no puede capturar la variabilidad real del problema.

#  Con el dataset mas amplio
Con el cambio del dataset que incluye mas datos, el modelo está funcionando bien o mucho mejor, con una alta precisión, aunque llama la atención que la clasificación de "Bajo Riesgo" tiene una ligera caída en el recall (0.86), lo que puede dar a entender que algunos casos de bajo riesgo fueron mal clasificados.
"""