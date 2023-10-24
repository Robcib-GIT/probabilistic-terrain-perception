import tensorflow as tf
import json
import numpy as np
import cv2

# Se inicializan las variables globales
img_arrays = [] # Guarda las imágenes en formato vector 
y_train = []    # Guarda las coordenadas que servirán como entrenamiento (cinco coordenadas por imagen)

# Se cargan las coordenadas y las imágenes, que servirán para entrenar la red
with open('dataset.json', 'r') as f:
    for linea in f:
        coordenadas = json.loads(linea)                         # Se cargan las coordenadas del json
        y_imagen = []                                           # Guarda las coordenadas de una imagen 
        for j, coord in enumerate(coordenadas['coordenadas']):
            y_imagen.append(coord['center'])
        y_train.append(y_imagen)                                # Añadimos el vector de coordenadas de una imagen al vector de vectores de coordenadas

        # Se carga la imagen determinada
        i = coordenadas['coordenadas'][0]['id']
        img_paths= [f'/home/robcib/Escritorio/TFG_Miguel/dataset/mascara_{i}.png']
        # Se lee la imagen y se convierte al formato que interesa para entrenar la red
        for path in img_paths:
            img = cv2.imread(path)
            img_array = np.array(img)
            img_array = cv2.resize(img_array, (640, 640)) 
            img_array = img_array.reshape((640, 640, 3))
            img_arrays.append(img_array)                    # Se añade al vector de imágenes

# Se ponen en el formato adecuado
img_arrays=np.array(img_arrays)
y_train=np.array(y_train)

# Se define la arquitectura de la red neuronal
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(640, 640, 3)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(5 * 2, activation='linear'),
    tf.keras.layers.Reshape((5, 2))
    ])

# Se compila el modelo
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
metricas = model.compile(optimizer=opt,
              loss='mean_squared_error',
              metrics=['accuracy'])

# Se entrena el modelo
model.fit(img_arrays, y_train, epochs=500)

model.save("modelov2.h5")

