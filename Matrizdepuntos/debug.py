from ultralytics import YOLO
import cv2
import numpy as np
import segmentacion
import json
import os

# Se carga el modelo entrenado de YOLO
model= YOLO("/home/robcib/Escritorio/TFG_Miguel/Entrenamiento_definitivo/weights/best.pt")

# Se carga el vídeo sobre el que se van a correr las predicciones
capture = cv2.VideoCapture("/home/robcib/Escritorio/Videos Yolo v8/ent.mp4")

# Se inicializan las variables globales
current_frame = 0   # Caracteriza y diferencia cada imagen 
colors = []         # Guarda el código de colores
labels = []         # Guarda los nombres de las etiquetas

# Se inicializa el objeto VideoWriter para guardar el video

fps = 30  # Fotogramas por segundo del video de salida
fourcc = cv2.VideoWriter_fourcc(*'xvid')  # Códec de video para el archivo de salida
out = cv2.VideoWriter("video.avi", fourcc, fps, (640, 480))  # Ajusta el tamaño (640x480) según tus preferencias


while (capture.isOpened()):
    
    # Se toman capturas del vídeo (a una frecuencia de 30 fotogramas/segundo)
    ret, frame = capture.read()

    if ret:

        # Se guarda y se carga la imagen en formato array, ya que es necesario para
        # poder realizar las operaciones que necesitamos con ella
        cv2.imwrite(f'frame_{current_frame}.jpg', frame) 
        image = cv2.imread(f'frame_{current_frame}.jpg')
        image = np.asarray(image)

        # Se cargan las predicciones
        results=model.predict(image)

        # Se cargan y adaptan las máscaras de segmentación a la imagen. Se añade la estructura 
        # try - except para que no falle el código en el que caso de que no haya detecciones
        try:
            masks = results[0].masks.data.cpu().numpy()
            masks = np.moveaxis(masks, 0, -1)
            masks = cv2.resize(masks, (results[0].masks.orig_shape[1], results[0].masks.orig_shape[0]))
            masks = np.moveaxis(masks, -1, 0)
        except:
           continue
        
        # Se establecen el código de colores y la etiqueta que se le asigna a cada deteccion
        if colors == []:
            colors = [(0,0,0),(255,150,0),(0,0,200),(26,127,239),(128,128,128),(0,255,0),(255,255,255),(42,42,165)]
        if labels == []: 
            labels = {1: u'NS', 2: u'S100', 3: u'S50',4: u'S75', 5: u'cemento', 6: u'cesped' , 7: u'grava', 8: u'tierra'}
        
        # Se inicializan algunas variables y se crean otras como constantes
        image_with_masks = np.copy(image)   # Se realiza una copia de la imagen original para superponerla con las máscaras
        i = 0                               # Representa el número de detecciones por imagen
        distancia = 30                      # Representa la distancia entre puntos de la matriz de puntos
        size = 10                           # Representa el tamaño de los puntos
        puntos = []                         # Variable en la que se guardan los puntos dibujados
        p = []
        conf = 0.6                          # Confianza que sirve de filtro para mostrar por pantalla las predicciones
        flag = 0
       
        # Se dibujan tanto las máscaras de segmentación como la matriz de puntos en la imagen. De nuevo, la estructura
        # try - except se emplea como prevención para que el código no falle inesperadamente
        try:
            for mask_i in masks:
                if results[0].boxes.data[i][4] > conf:
                    image_with_masks = segmentacion.overlay(image_with_masks, mask_i,  color=(255,255,255), alpha=0.2)
                    puntos.append(segmentacion.dibujapuntos(image_with_masks, mask_i , distancia, size, i, labels, puntos, results[0].boxes.data[i]))
                    p = puntos       
                i = i + 1
        except:
           continue
        
        # Se dibujan las cajas delimitadoras para cada predicción en la imagen
        for b in results[0].boxes.data:
            # Se asigna a cada caja su correspondiente etiqueta
            try:
                label = labels[int(b[-1])+1] 
            except:
                break
            # Se dibujan las bounding boxes si la confianza que devuelve el modelo es mayor que la que 
            # utilizamos como valor umbral
            if conf :
                if b[-2] > conf:
                    color = colors[int(b[-1])]
                    flag = segmentacion.box_label(image_with_masks, b, flag, label, color)
            else:
                color = colors[int(b[-1])]
                flag = segmentacion.box_label(image_with_masks, b, flag, label, color)
        
        puntos.append(segmentacion.entorno(image_with_masks,distancia,size,flag,puntos))
        p = puntos
        # Se guardan las coordenadas de los puntos pertenecientes a la matriz de puntos en cada imagen
        # con el objetivo de crear un dataset para entrenar la red
        data = {}
        data[f"puntos"] = p
        #with open("matrizpuntostrash.json", "a") as f:
            #json.dump(data, f)
            #f.write('\n')
        
        # Se borran las imágenes que no se necesitan para no ocupar espacio de manera innecesaria
        if os.path.exists(f'frame_{current_frame}.jpg'):
            os.remove(f'frame_{current_frame}.jpg')

        # Se guarda la imagen con todas las predicciones que se han añadido para entrenar la red
        #cv2.imwrite(f'mascaratrash_{current_frame}.png', image_with_masks)

        # Se muestra por pantalla el resultado
        cv2.namedWindow('matrizpuntos', cv2.WINDOW_NORMAL)
        cv2.imshow('matrizpuntos',image_with_masks) 
        cv2.resizeWindow('matrizpuntos', 640, 480)
        cv2.waitKey(1)
        out.write(image_with_masks)

        # Se preparan las variables para una nueva imagen
        current_frame = current_frame + 1

    else:
        break

# Al acabar el vídeo, se liberan y se destruyen todas las ventanas creadas
capture.release()
out.release()
cv2.destroyAllWindows() 