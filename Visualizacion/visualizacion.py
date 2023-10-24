from ultralytics import YOLO
import cv2
import numpy as np
import segmentacion
import os

# Se carga el modelo entrenado de YOLO
model= YOLO("/home/robcib/Escritorio/TFG_Miguel/Entrenamiento_definitivo/weights/best.pt") 

# Se carga el vídeo sobre el que se van a correr las predicciones
capture = cv2.VideoCapture("/home/robcib/Escritorio/Videos Yolo v8/ent.mp4")

# Se inicializan las variables globales
current_frame = 0   # Caracteriza y diferencia cada imagen 
colores = []        # Guarda el código de colores
etiquetas = []      # Guarda los nombres de las etiquetas

# Se entra en el bucle infinito, y no se sale de él hasta que acabe el vídeo usado para las predicciones
while (capture.isOpened()):
    # Se toman capturas del vídeo (a una frecuencia de 30 fotogramas/segundo)
    recepcion, frame = capture.read()

    if recepcion:
        # Se guarda y se carga la imagen en formato vector, ya que es necesario para
        # poder realizar las operaciones que se necesitan con ella
        cv2.imwrite(f'frame_{current_frame}.jpg', frame)    
        imagen = cv2.imread(f'frame_{current_frame}.jpg')
        imagen = np.asarray(imagen)

        # Se cargan las predicciones
        resultados = model.predict(imagen)

        # Se cargan y adaptan las máscaras de segmentación a la imagen. Se añade la estructura 
        # try - except para que no falle el código en el que caso de que no haya detecciones
        try:
            mascaras = resultados[0].masks.data.cpu().numpy()
            mascaras = np.moveaxis(mascaras, 0, -1)
            mascaras = cv2.resize(mascaras, (resultados[0].masks.orig_shape[1], resultados[0].masks.orig_shape[0]))
            mascaras = np.moveaxis(mascaras, -1, 0)
        except:
            continue
        
        # Se establecen el código de colores y la etiqueta que se le asigna a cada deteccion
        if colores == []:
            colores = [(0,0,0),(255,150,0),(0,0,200),(26,127,239),(128,128,128),(0,255,0),(255,255,255),(42,42,165)]
        if etiquetas == []: 
            etiquetas = {1: u'NS', 2: u'S100', 3: u'S50',4: u'S75', 5: u'cemento', 6: u'cesped' , 7: u'grava', 8: u'tierra'}
        
        # Se inicializan algunas variables y se crean otras como constantes
        copia_imagen = np.copy(imagen)      # Se realiza una copia de la imagen original para superponerla con las máscaras
        i = 0                               # Representa el número de detecciones por imagen
        conf = 0.6                          # Confianza que sirve de filtro para mostrar por pantalla las predicciones

        # Se dibujan tanto las máscaras de segmentación como la matriz de puntos en la imagen. De nuevo, la estructura
        # try - except se emplea como prevención para que el código no falle inesperadamente
        try:
            for mascara_i in mascaras:
                if resultados[0].boxes.data[i][4] > conf:
                    copia_imagen = segmentacion.dibujamascara(copia_imagen, mascara_i,  color=(255,255,255), alpha=0.5)
                i = i + 1
        except:
            continue

        # Se dibujan las cajas delimitadoras para cada predicción en la imagen
        for b in resultados[0].boxes.data:
            # Se asigna a cada caja su correspondiente etiqueta
            try:
                label = etiquetas[int(b[-1])+1] 
            except:
                break
            # Se dibujan las bounding boxes si la confianza que devuelve el modelo es mayor que la que 
            # utilizamos como valor umbral
            if conf :
                if b[-2] > conf:
                    color = colores[int(b[-1])]
                    segmentacion.dibujacajas(copia_imagen, b, label, color)
            else:
                color = colores[int(b[-1])]
                segmentacion.dibujacajas(copia_imagen, b, label, color)
        
        # Se borra la imagen guardada, por temas de espacio de almacenamiento
        if os.path.exists(f'frame_{current_frame}.jpg'):
            os.remove(f'frame_{current_frame}.jpg')

        # Se muestra por pantalla la imagen con todas las predicciones que se han añadido
        cv2.namedWindow('visualizacion', cv2.WINDOW_NORMAL)
        cv2.imshow('visualizacion', copia_imagen)
        cv2.resizeWindow('visualizacion', 640, 480)
        cv2.waitKey()
        
        # Se identifica de manera unívoca cada imagen
        current_frame = current_frame + 1

    else:
        break

# Al acabar el vídeo, se liberan y se destruyen todas las ventanas creadas
capture.release()
cv2.destroyAllWindows() 