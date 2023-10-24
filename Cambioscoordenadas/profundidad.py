import cv2                  # Necesaria para mostrar por pantalla las imágenes
import numpy as np          # Necesaria para el tratamiento de datos (imágenes en este caso)
import pyrealsense2 as rs   # Necesaria para la configuración de la cámara
import math                 # Necesaria para funciones como senos y cosenos

# Esta funcion convierte un píxel de la imagen, mediante la coordenada de profundidad, a coordenadas tridimensionales, 
# siendo el origen del sistema de referencia el sensor de profundidad de la cámara especificado en el documento
def calculocoord(parametros, pixel, prof_media, factor_escala):
    # Se calculan las coordenadas tridimensionales, de acuerdo a las ecuaciones descritas
    px = (pixel[0] - parametros.ppx) 
    py = (pixel[1] - parametros.ppy) 
    xs = (px * prof_media) / parametros.fx
    ys = (py * prof_media) / parametros.fy
    zs = prof_media * factor_escala
    return xs, ys, zs

# Función que calcula los cambios de base por traslación
def traslacion(x, y, z, a, b, c):
    x1 = x - a
    y1 = y - b
    z1 = z - c
    return x1, y1, z1

# Función que calcula los cambios de base por rotación
def rotacion(xc, yc, zc, Xrot, Yrot, Zrot):
    resul = np.array([xc, yc, zc])
    aux1 = np.dot(Xrot, resul)
    aux2 = np.dot(Yrot, aux1)
    resul = np.dot(Zrot, aux2)
    return resul

# Se realiza la configuración del pipeline
pipeline = rs.pipeline()
configuracion = rs.config()

# Se habilitan los canales por los que va a fluir la información
configuracion.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
configuracion.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Sentencias necesarias para obtener el factor de escala de profundidad de la cámara
contexto = rs.context()
lectura = contexto.query_devices()
sensorprof = lectura[0].first_depth_sensor()
factor_escala = sensorprof.get_depth_scale() 

# Se habilita el canal que permite el flujo de información (pipeline)
pipeline.start(configuracion)

# Se obtienen los parámetros intrínsecos de la cámara, guardados en la variable "parametros"
activacion = pipeline.get_active_profile()
camara = rs.video_stream_profile(activacion.get_stream(rs.stream.depth))
parametros = camara.get_intrinsics()

try:
    while True:
        # Se espera a la llegada de uana imagen a través del pipeline
        imagen = pipeline.wait_for_frames()

        # Se obtiene una imagen RGB y se 
        imagencolor = imagen.get_color_frame()

        # Se obtiene el frame de la imagen de profundidad
        imagenprofundidad = imagen.get_depth_frame()

        # Se definen las variables de iteración y las variables auxiliares
        xp = 100             # Coordenada x del punto (Variables de prueba para comprobar la efectividad de la lógica implementada)
        yp = 200             # Coordenada y del punto
        pixel = (xp,yp)
        itx = xp - 2         # Variables de iteración
        ity = yp - 2 
        i = 0               # Variable que cuenta el número de iteraciones
        prof_total = 0      # Variable que guarda el valor de la suma absoluta de distancias

        # Con este bucle se pretende darle robustez al método de obtención de la distancia de un punto de la imagen, ya que
        # no se ha realizado una calibración previa a su utilización.
        # Además, esta decisión también se ha tomado teniendo en cuenta la sensibilidad especial que tienen las cámaras de 
        # profundidad a las luces y sombras. El número de puntos elegido se ha determinado empíricamente, cuando se ha visto que 
        # la precisión dejaba de variar de manera significativa
        for m in range(5):
            for n in range(5):
                prof = imagenprofundidad.get_distance(itx, ity)  # Se obtiene el valor en profundidad del píxel 
                prof_total += prof                               # Se suman las profundidas
                i += 1                                           # Lógica de iteración
                itx += 1
            itx = xp - 2
            ity += 1
        
        # Se calcula el valor medio de las distancias de profundidad
        prof_media = prof_total/i  

        # Se convierte la coordenada de profundidad a coordenadas tridimensionales
        xs, ys, zs = calculocoord(parametros, pixel, prof_media, factor_escala)

        # Parámetros de traslación entre sistema del sensor y sistema de la cámara (en m). 
        # Variarán en función del robot que se esté empleando
        a = -0.0175
        b = -0.0125
        c = -0.0042

        # Traslación del sistema de referencia del sensor al de la cámara
        xc, yc, zc = traslacion(xs, ys, zs, a, b, c)

        # Ángulos de rotación, en radianes entre coordenadas del sistema de la cámara 
        # y coordenadas del sistema de referencia global particulares para este caso
        gamma = 0
        theta = -(math.pi)/2
        beta = 0

        # Se definen las matrices de rotación de manera general, para cualquier distribución de ejes
        Zrot =np.array ([
            [math.cos(gamma), -math.sin(gamma), 0],
            [math.sin(gamma), math.cos(gamma), 0],
            [0, 0, 1]
        ])

        Yrot = np.array ([
            [math.cos(theta), 0, math.sin(theta)],
            [0, 1, 0],
            [- math.sin(theta), 0, math.cos(theta)]
        ])

        Xrot = np.array ([
            [1, 0, 0],
            [0, math.cos(beta), -math.sin(beta)],
            [0, math.sin(beta), math.cos(beta)]
        ])

        # Se rota el sistema de coordenadas frente al del robot
        R = rotacion(xc, yc, zc, Xrot, Yrot, Zrot)

        # Parámetros de traslación entre sistema del sensor y sistema de la cámara (en mm)
        a = - 0.25
        b = 0
        c = 0

        # Se traslada el punto desde el sistema de coordenadas de la cámara al robot
        x, y, z = traslacion(R[0], R[1], R[2], a, b, c)

        # Se imprimen por pantalla las coordenadas del punto escogido en 2D y su valor en profundidad
        print("Profundidad en el punto ({},{}): {} unidades".format(x, y, prof_media))

        # Se imprimen por pantalla las coordenadas (x,y,z) del punto respecto del sistema de coordenadas global
        print("Coordenadas tridimensionales (x, y, z) respecto del sistema de coordenadas global:", pixel)
        print("x:", x)
        print("y:", y)
        print("z:", z)

        # Se adecúan las imágenes al formato que se necesita para que lo procese la librería OpenCV
        color_image = np.asanyarray(imagencolor.get_data())
        depth_image = np.asanyarray(imagenprofundidad.get_data())

        # Se cambia la tonalidad del mapa de profundidad para aclarar y facilitar su análisis
        depth_image_coloreada = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        
        # Se muestran las imágenes RGB y de profundidad respectivamente
        cv2.imshow('Imagen RGB', color_image)
        cv2.imshow('Imagen de profundidad', depth_image_coloreada)
        cv2.waitKey()
        


finally:
    # Se libera el canal de información y se destruyen las ventanas creadas
    pipeline.stop()
    cv2.destroyAllWindows()