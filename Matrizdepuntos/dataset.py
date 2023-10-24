import cv2
import json
import numpy as np

#Se inicializan las variables globales
k = 0               # Sirve para diferenciar las imágenes entre sí
coordenadas = []    # Guarda las coordenadas que se eligen como etiquetas

# Función que se encarga de detectar el evento del ratón en la ventana. 
# Esta es la forma en la que se realizan las anotaciones para el entrenamiento de 
# esta nueva red neuronal
def detectar_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        # Una vez detectado el evento del ratón, recorre todos los puntos de la imagen
        # para ver a cuál de ellos corresponde. Esto lo hace calculando la distancia entre
        # el punto donde se hace click y la posición de cada punto
        for p in puntos:
            if (x - p[0])**2 + (y - p[1])**2 <= 20:
                coordenadas.append({"id": k, "center": [p[0], p[1]]})

# Se cargan tanto las coordenadas de los puntos de la imagen como la propia imagen,
# para ser capaces de realizar las anotaciones
with open('matrizpuntos.json', 'r') as archivo:
    # En cada fila del archivo json están las coordenadas de cada imagen
    for linea in archivo:
        datos = json.loads(linea)
        i = datos['puntos'][0][0]['id']
        img = cv2.imread(f'/home/robcib/Escritorio/TFG_Miguel/dataset/mascara_{k}.png')
        
        puntos = np.empty((0, 2))       # Se inicializa la variable donde se guardarán los puntos
        
        # Se preparan los datos para realizar las anotaciones
        pequeño = len(datos['puntos']) 
        # Se guardan todos los puntos de cada línea del json en la variable puntos 
        for m in range(pequeño):
            grande = range(len(datos['puntos'][m]))
            for n in grande:
                punto = np.array(datos['puntos'][m][n]['center'])
                puntos = np.vstack([puntos, punto])

        # Se muestran las imagenes escaladas por pantalla y se espera al click del ratón
        cv2.namedWindow('Imagen con puntos', cv2.WINDOW_NORMAL)
        cv2.imshow('Imagen con puntos', img)
        cv2.resizeWindow('Imagen con puntos', 1080, 1920)
        cv2.setMouseCallback('Imagen con puntos', detectar_click)
        
        # Se guardan las anotaciones en un nuevo archivo json
        c = {}
        c[f"coordenadas"] = coordenadas
        if coordenadas != []:
            with open("dataset.json", "a") as f:
                json.dump(c, f)
                f.write('\n')
        
        # Se preparan las variables para la siguiente imagen
        coordenadas.clear()
        k = k + 1

        cv2.waitKey()

cv2.destroyAllWindows()
   

