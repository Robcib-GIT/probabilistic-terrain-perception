import tensorflow as tf
import json
import numpy as np
import cv2

# Se carga el modelo entrenado
modelo = tf.keras.models.load_model("modelo.h5")

# Se inicializan las variable globales
k = 0           # Variable que se utiliza para identificar unívocamente una imagen
centros = []    # Variable que guarda los puntos ya predichos 

# Función que se encarga de asociar el punto predicho a un punto que pertenezca
# a la matriz de puntos generada en la imagen
def proximidad(punto, coordenadas, i):
    num_detec = len(datos['puntos'])    # Número de puntos de la imagen
    centro_mas_cercano = None           # Inicializa la variable donde se guarda el punto más cercano
    distancia_minima = float('inf')     # Se inicializa la variable donde se guarda la distancia entre puntos
    # Se recorre toda la matriz de puntos de la imagen en busca del punto de la matriz más próximo al punto predicho
    for j in range(num_detec):
        for zona in coordenadas['puntos'][j]:
            centro_zona = tuple(zona['center'])
            distancia = ((punto[0]-centro_zona[0])**2 + (punto[1]-centro_zona[1])**2)**0.5
            if distancia < distancia_minima:
                if centro_zona in centros:          # Se comprueba si el punto ya ha sido elegido, en cuyo caso no se añade
                    break                           # para evitar solapamientos
                distancia_minima = distancia
                centro_mas_cercano = centro_zona
    centros.append(centro_mas_cercano)              # Se añade el punto predicho a la lista para descartarlo en caso de que 
                                                    # vuelva a ser elegido
    return centro_mas_cercano

with open('matrizpuntos.json', 'r') as archivo:
    for linea in archivo:
        # Variable auxiliar que lleva cuenta del número de puntos
        i = 0 
        # Se carga la matriz de puntos y su imagen correspondiente
        datos = json.loads(linea)
        try:
            img = cv2.imread(f'/home/robcib/Escritorio/TFG_Miguel/dataset/mascara_{k}.png')
            img = np.array(img)
            img_array = cv2.resize(img, (640, 640))
            img_array = np.reshape(img_array, (1,640, 640,3))
        except:
            continue

        # Se predicen las coordenadas y se escalan de manera adecuada
        prediccion = modelo.predict(img_array)
        prediccion = prediccion.reshape(-1, 2)
        prediccion = [(x, y) for x, y in prediccion]
        
        # Se recalculan esos puntos predichos para que se aproximen a un punto existente en la matriz de puntos
        for j in range(5):
            punto_predicho = prediccion[j]
            punto_predicho = (int(punto_predicho[0]),int(punto_predicho[1]))
            punto = proximidad(punto_predicho, datos, i)
            # Elección de colores en función del punto que sea
            if i == 0:
                cv2.circle(img, punto, 10, (255, 255, 255), -1) # Blanco
            elif i == 1:
                cv2.circle(img, punto, 10, (200, 200, 200), -1) # Verde claro
            elif i == 2:
                cv2.circle(img, punto, 10, (150, 150, 150), -1) # Verde normal
            elif i == 3:
                cv2.circle(img, punto, 10, (100, 100, 100), -1) # Verde oscuro
            elif i == 4:
                cv2.circle(img, punto, 10, (50, 50, 50), -1)    # Gris
            i = i + 1

        # Se preparan las variables para la siguiente predicción
        centros.clear()
        k = k + 1

        # Se muestra por pantalla el resultado
        cv2.namedWindow('prediccion', cv2.WINDOW_NORMAL)
        cv2.imshow('prediccion',img) 
        cv2.resizeWindow('prediccion', 1080, 1920)
        cv2.waitKey()
