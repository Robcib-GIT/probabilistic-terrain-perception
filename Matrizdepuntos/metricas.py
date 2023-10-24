import tensorflow as tf
import json
import numpy as np
import cv2
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

modelo = tf.keras.models.load_model("modelo.h5")
k = 0 

y_train = [] 
y_preds = []
y = []

with open('dataset.json', 'r') as f:
    for linea in f:
        coordenadas = json.loads(linea)                         # Se cargan las coordenadas del json
        y_imagen = []                                           # Guarda las coordenadas de una imagen 
        for j, coord in enumerate(coordenadas['coordenadas']):
            y_imagen.append(coord['center'])
        y_train.append(y_imagen)                                # Añadimos el vector de coordenadas de una imagen al vector de vectores de coordenadas

        # Se carga la imagen determinada
        i = coordenadas['coordenadas'][0]['id']


y_train = np.array(y_train)

with open('matrizpuntos.json', 'r') as archivo:
    for linea in archivo:
        # Se carga la matriz de puntos y su imagen correspondiente
        datos = json.loads(linea)
        try:
            img = cv2.imread(f'/home/robcib/Escritorio/TFG_Miguel/dataset/mascara_{k}.png')
            img = np.array(img)
            img_array = cv2.resize(img, (640, 640))
            img_array = np.reshape(img_array, (1,640, 640,3))
        except:
            continue

        y_pred = modelo.predict(img_array)
        y_pred = np.array(y_pred)
        y_preds.append(y_pred)
        k = k + 1
        if k==262:
            break

y_preds = np.squeeze(y_preds).astype(int)
y_train = y_train.astype(int)
y_preds = y_preds.flatten()
y_train = y_train.flatten()
num_predicciones = len(y_train)
print(num_predicciones)

for i in range(num_predicciones):
    resta = abs(y_train[i] - y_preds[i])
    y.append(resta)    


#fig, eje = plt.subplots()
#x = np.linspace(0, 1, num_predicciones)
#eje.bar(x,y)
#plt.show()
x = np.arange(num_predicciones)  # Valores x para las barras (números de 0 a num_predicciones-1)
plt.bar(x, y)

plt.xlabel('Número de predicción')
plt.ylabel('Coordenadas en pixeles')
plt.title('Diferencias entre valores reales y predichos')

plt.show()
#"""
mse = mean_squared_error(y_train, y_preds)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_train, y_preds)
r2 = r2_score(y_train, y_preds)

# Imprimir las métricas
print("Error cuadrático medio por punto (MSE):", mse)
print("Raíz del error cuadrático medio (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("R^2 Score:", r2)
#"""

plt.plot(y_train, 'o', label='Valores reales', color = 'green')
plt.plot(y_preds,  'o', label='Valores predichos', color = 'red')
plt.xlabel('Número de predicción')
plt.ylabel('Coordenadas (en pixeles)')
plt.title('Comparación entre valores reales y predichos')
plt.legend()

# Mostrar la gráfica
plt.show()


