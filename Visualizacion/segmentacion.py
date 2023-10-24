import numpy as np
import cv2

# Función que se encarga de combinar la imagen original con la imagen con máscara de segmentación, 
# para así incluir esta predicción en la imagen leída desde la cámara
def dibujamascara(image, mask, color, alpha):
    color = color[::-1]                                                                     # Se invierte el orden de los colores de BGR a RGB
    mascara = np.expand_dims(mask, 0).repeat(3, axis=0)                                     # Se adecúa las dimensiones de la máscara a la imagen
    mascara = np.moveaxis(mascara, 0, -1)                                 
    enmascarado = np.ma.MaskedArray(image, mascara, fill_value=color)                       # Se aplica la máscara
    image_superpuesta = enmascarado.filled()                                                # Se recupera la imagen con la máscara
    image_enmascarada = cv2.addWeighted(image, 1 - alpha, image_superpuesta, alpha, 0)      # Se combina imagen original con imagen enmascarada
    return image_enmascarada

# Función que se encarga de dibujar las cajas delimitadoras predichas en la imagen
def dibujacajas(image, b, etiq='', color=(128, 128, 128), txt_color=(200, 200, 200)):

    anchura = max(round(sum(image.shape) / 2 * 0.003), 2)                           # Anchura de la línea de la caja delimitadora
    p1, p2 = (int(b[0]), int(b[1])), (int(b[2]), int(b[3]))                         # Coordenadas de la caja
    cv2.rectangle(image, p1, p2, color, thickness=anchura, lineType=cv2.LINE_AA)    # Se dibuja la caja
    etiq = etiq + " " + str(round(100 * float(b[-2]),1)) + "%"                      # Se añade la confianza de la predicción a la etiqueta
    if etiq: 
        grosor_letra = max(anchura - 1, 1)                                                    # Grosor de la fuente 
        ancho, alto = cv2.getTextSize(etiq, 0, fontScale=anchura / 3, thickness=grosor_letra)[0]     # Anchura y altura del texto
        outside = p1[1] - alto >= 3                                                    
        p2 = p1[0] + ancho, p1[1] - alto - 3 if outside else p1[1] + alto + 3                 # Se decide si se escribe el texto dentro o fuera de la caja
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)                        # Se dibuja la caja y se escribe la etiqueta
        cv2.putText(image, etiq, (p1[0], p1[1] - 2 if outside else p1[1] + alto + 2), 0, 
                    anchura/3, txt_color, thickness=grosor_letra, lineType=cv2.LINE_AA)
  