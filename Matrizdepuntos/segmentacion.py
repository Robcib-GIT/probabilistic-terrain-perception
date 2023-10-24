import numpy as np
import cv2

# Se inicializan la variable global
dibujado = []   # Guarda los puntos ya dibujados, para que no haya solapamientos

# Función que se encarga de dibujar la matriz de puntos en cada imagen
def dibujapuntos(image, mask, d, size, j, labels, puntos, b):  

    # Se definen e inicializan todas las variables necesarias   
    label = labels[int(b[-1])+1]    # Representa la etiqueta de la predicción
    y_max = image.shape[0]          # Altura de la imagen
    ytercios = int(0.33*y_max)
    ydostercios = int(0.67*y_max)
    x_max = image.shape[1]          # Anchura de la imagen
    l1_min = int(b[0])              # Coordenadas de las esquinas de la caja delimitadora
    l2_min = int(b[1])              
    l1_max = int(b[2])              
    l2_max = int(b[3])              
    y = 0                           # Variables de iteración
    x = 0                           

    # Se recorre toda la imagen mientras se dibujan los puntos de acuerdo a los criterios establecidos 
    while 0 <= y < y_max:
        if y < ytercios:
            size = int(0.8*size)
            d = int(0.8*d)
        if y < ydostercios:
            size = int(0.8*size)
            d = int(0.8*d)
        while 0 <= x < x_max:
            if (l1_min < x < l1_max) and (l2_min < y < l2_max):
                # Solo dibujamos el punto si este no ha sido dibujado antes
                if (x,y) not in dibujado or label=='S100' or label=='S75' or label=='S50' or label=='NS':
                    # Si el punto pertenece a la máscara de segmentación se sigue un criterio de colores
                    if mask[y][x] != 0:
                        if label=='NS': 
                            cv2.circle(image, (x,y), size, (0,0,0), -1)
                            puntos.append({"id": j, "center": [x, y], "radius": size})
                            dibujado.append((x,y))
                        elif label=='S50': 
                            cv2.circle(image, (x,y), size, (0,0,200), -1)
                            puntos.append({"id": j, "center": [x, y], "radius": size})
                            dibujado.append((x,y))
                        elif label=='S75': 
                            cv2.circle(image, (x,y), size, (26,127,239), -1)
                            puntos.append({"id": j, "center": [x, y], "radius": size})
                            dibujado.append((x,y))
                        elif label=='S100': 
                            cv2.circle(image, (x,y), size, (255,150,0), -1)
                            puntos.append({"id": j, "center": [x, y], "radius": size})
                            dibujado.append((x,y))
                        elif label=='grava': 
                            cv2.circle(image, (x,y), size, (255,255,255), -1)
                            puntos.append({"id": j, "center": [x, y], "radius": size})
                            dibujado.append((x,y))
                        elif label=='cemento': 
                            cv2.circle(image, (x,y), size, (128,128,128), -1)
                            puntos.append({"id": j, "center": [x, y], "radius": size})
                            dibujado.append((x,y))
                        elif label=='tierra': 
                            cv2.circle(image, (x,y), size, (42, 42, 165), -1)
                            puntos.append({"id": j, "center": [x, y], "radius": size})
                            dibujado.append((x,y))
                        else:            
                            cv2.circle(image, (x,y), size, (0,255,0), -1)
                            puntos.append({"id": j, "center": [x, y], "radius": size})
                            dibujado.append((x,y))
                            # Si el punto pertenece a la caja delimitadora pero no a la máscara, se sigue otro criterio
                    else:
                        if label=='NS': 
                            cv2.circle(image, (x,y), size, (0,0,200), -1)
                            puntos.append({"id": j, "center": [x, y], "radius": size})
                            dibujado.append((x,y))
                        elif label=='S50': 
                            cv2.circle(image, (x,y), size, (255,75,0), -1)
                            puntos.append({"id": j, "center": [x, y], "radius": size})
                            dibujado.append((x,y))
                        elif label=='S75': 
                            cv2.circle(image, (x,y), size, (255,150,0), -1)
                            puntos.append({"id": j, "center": [x, y], "radius": size})
                            dibujado.append((x,y))
                        elif label=='S100':            
                            cv2.circle(image, (x,y), size, (255,220,0), -1)
                            puntos.append({"id": j, "center": [x, y], "radius": size})
                            dibujado.append((x,y))                    
            x += d
        x = d
        y += d
        size = 10
        d = 30
    return puntos


# Función que se encarga de combinar la imagen original con la imagen con máscara de segmentación, 
# para así incluir esta predicción en la imagen leída desde la cámara
def overlay(image, mask, color, alpha):
    color = color[::-1]                                                             # Se invierte el orden de los colores de BGR a RGB
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)                        # Se adecúa las dimensiones de la máscara a la imagen
    colored_mask = np.moveaxis(colored_mask, 0, -1)                                 
    masked = np.ma.MaskedArray(image, colored_mask, fill_value=color)               # Se aplica la máscara
    image_overlay = masked.filled()                                                 # Se recupera la imagen con la máscara
    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)     # Combina imagen original con imagen enmascarada
    return image_combined

# Función que se encarga de dibujar las cajas delimitadoras predichas en la imagen
def box_label(image, b, flag, label='', color=(128,128,128), txt_color=(200,200,200)):
    # Si ya ha habido detección de terreno, se indica aquí para no entrar en la siguiente función
    if label=='cemento' or label=='cesped' or label == 'grava' or label == 'cemento':
        flag = 1
    anchura = max(round(sum(image.shape) / 2 * 0.003), 2)                           # Anchura de la línea de la caja delimitadora
    p1, p2 = (int(b[0]), int(b[1])), (int(b[2]), int(b[3]))                         # Coordenadas de la caja
    cv2.rectangle(image, p1, p2, color, thickness=anchura, lineType=cv2.LINE_AA)    # Se dibuja la caja
    label = label + " " + str(round(100 * float(b[-2]),1)) + "%"                    # Se añade la confianza de la predicción a la etiqueta
    if label: 
        tf = max(anchura - 1, 1)                                                    # Grosor de la fuente 
        w, h = cv2.getTextSize(label, 0, fontScale=anchura / 3, thickness=tf)[0]    # Anchura y altura del texto
        outside = p1[1] - h >= 3                                                    
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3                 # Decide si escribe el texto dentro o fuera de la caja
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)                        # Se dibuja la caja y se escribe la etiqueta
        cv2.putText(image, label, (p1[0], p1[1]-2 if outside else p1[1]+h+2), 0, 
                    anchura/3, txt_color, thickness=tf, lineType=cv2.LINE_AA)
    return flag

# Función que se encarga de dibujar la matriz de puntos por defecto en el caso de que no hayan habido detecciones
def entorno(image, d, size, flag, puntos):
    # Si ha habido detección de terreno, no se ejecuta esta función
    if flag == 1:
        dibujado.clear()
        return puntos
    elif flag == 0:      
        y_max = image.shape[0]
        ytercios = int(0.33*y_max)
        ydostercios = int(0.67*y_max)
        x_max = image.shape[1]
        y = 0                           
        x = 0  
        while 0 <= y < y_max:
            if y < ytercios:
                size = int(0.8*size)
                d = int(0.8*d)
            if y < ydostercios:
                size = int(0.8*size)
                d = int(0.8*d)
            while 0 <= x < x_max:
                if (x,y) not in dibujado:
                    cv2.circle(image, (x,y), size, (0,255,255), -1)
                    puntos.append({"id": 0, "center": [x, y], "radius": size})
                    dibujado.append((x,y))
                x += d
            x = d
            y += d
            size = 10
            d = 30
        dibujado.clear()
        return puntos