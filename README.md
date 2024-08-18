# Proyecto G6 - Detector de Imagenes de RayosX, con modelado de IA

Este proyecto consiste en una interfaz gráfica para la detección de neumonía utilizando un modelo de red neuronal convolucional (CNN) entrenado previamente. La aplicación permite cargar una imagen de rayos X, procesarla, y realizar una predicción sobre si el paciente presenta o no neumonía.

## Requisitos del Sistema

Este proyecto requiere Python 3.8 o superior. Las siguientes bibliotecas de Python deben estar instaladas:

- `tensorflow`
- `tkinter`
- `Pillow`
- `opencv-python`
- `numpy`
- `matplotlib`

Puedes instalar estas dependencias utilizando `pip`:

```bash
pip install tensorflow pillow opencv-python numpy matplotlib
```

## Estructura del Proyecto

- interfaz.py: Contiene el código principal de la interfaz gráfica y la lógica de predicción.
- mi_modelo.h5: Archivo del modelo preentrenado de TensorFlow utilizado para realizar las predicciones.


## Descripcion del Proceso 

1. Carga del Modelo: Al iniciar la aplicación, se carga el modelo preentrenado (mi_modelo.h5).

2. Carga de Imagen: El usuario puede cargar una imagen de rayos X desde su sistema local a través de la interfaz.

3. Preprocesamiento de la Imagen: La imagen se convierte a escala de grises, se redimensiona al tamaño esperado por el modelo, y se normaliza antes de pasarla a la red neuronal para la predicción.

4. Predicción: El modelo realiza una predicción sobre la imagen procesada. La predicción se muestra en la interfaz junto con su valor exacto (entre 0 y 1) y un gráfico de medición tipo gauge.

## Uso de la Aplicación
1. Ejecuta el archivo interfaz.py:

```bash
python interfaz.py
```

2. En la interfaz gráfica, haz clic en "Cargar Imagen" para seleccionar una imagen de rayos X desde tu computadora.

3. La aplicación mostrará la imagen cargada, la predicción del modelo, el valor de la predicción en el gráfico de medición tipo gauge, y los hiperparámetros del modelo.

## Consideraciones

- Formato de Imagen: El modelo espera que la imagen de entrada esté en escala de grises y tenga un tamaño de 150x150 píxeles.
- Rendimiento: El tiempo de predicción puede variar dependiendo del hardware y del tamaño de la imagen.