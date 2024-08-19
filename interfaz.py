import tkinter as tk
from tkinter import filedialog, Canvas, Scrollbar, HORIZONTAL, VERTICAL
from PIL import Image, ImageTk
import numpy as np
import cv2
from tensorflow.keras.models import load_model, Model
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Cargar modelo
model = load_model('mi_modelo.h5')

# Tamaño de la imagen que espera el modelo
img_size = 150

# Variables para manejar el zoom y desplazamiento
zoom_factor = 1.0
img_original = None
img_display = None
heatmap_phi = None
img_width, img_height = 0, 0

# Dimensiones del recuadro donde se mostrará la imagen
display_width = 400
display_height = 400


def generar_mapa_calor(img):
    global img_original
    img_array = np.asarray(img_original)
    class_weights = model.layers[-1].get_weights()[0]

    conv_layer_name = "conv2d_4"
    final_conv_layer = model.get_layer(conv_layer_name)

    get_output = Model([model.layers[0].input],
                       [final_conv_layer.output, model.layers[-1].output])

    [conv_outputs, predictions] = get_output(img)
    conv_outputs = conv_outputs[0, :, :, :]

    cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[0:2])

    for i, w in enumerate(class_weights[:, 0]):
        cam += w * conv_outputs[:, :, (i*2)+1]
        cam += w * conv_outputs[:, :, i*2]
    cam /= np.max(cam)
    final_cam = cv2.resize(cam.numpy(), (display_width, img_array.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255*final_cam), cv2.COLORMAP_JET)
    heatmap[np.where(final_cam < 0.1)] = 0
    # Convertir la imagen en escala de grises a una imagen RGB
    img_3channel = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

    # Realiza la suma con el mapa de calor
    final_result = cv2.add(heatmap, img_3channel)
    heatmap_ph = Image.fromarray(final_result, "RGB")
    img_original = heatmap_ph


# Función para procesar la imagen
def predecir_imagen(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_size, img_size))  # ajustar el tamaño
    img = img / 255.0  # normalización
    img = np.expand_dims(img, axis=0)  # dimensión de lote
    img = np.expand_dims(img, axis=-1)  # dimensión de canal

    prediccion = model.predict(img)[0][0]

    generar_mapa_calor(img)
    # normalizando valor para la escala de 0 a 1
    prediccion = min(max(prediccion, 0), 1)
    return prediccion


# Función para actualizar la imagen con zoom y desplazamiento
def actualizar_imagen_zoom():
    global img_display, zoom_factor

    if img_original:
        # Aplicar zoom
        new_width = int(img_width * zoom_factor)
        new_height = int(img_height * zoom_factor)

        # Limitar el tamaño del canvas a la imagen redimensionada
        canvas.config(scrollregion=(0, 0, new_width, new_height))

        img_zoom = img_original.resize((new_width, new_height), Image.LANCZOS)
        img_display = ImageTk.PhotoImage(img_zoom)

        # Limpiar el canvas antes de cargar la nueva imagen
        canvas.delete("all")
        canvas.create_image(0, 0, image=img_display, anchor=tk.NW)


# Cargar imagen a analizar
def cargar_imagen():
    global img_original, zoom_factor, img_width, img_height
    zoom_factor = 1.0  # Reiniciar zoom al cargar una nueva imagen

    file_path = filedialog.askopenfilename()
    img_original = Image.open(file_path)
    img_width, img_height = img_original.size

    # Redimensionar la imagen para ajustarla al ancho del canvas,
    # ajustando la altura proporcionalmente
    rf = img_width / display_width
    img_height_resized = int(img_height / rf)
    img_original = img_original.resize((display_width, img_height_resized))
    img_width, img_height = img_original.size

    # Ajustar la altura del canvas a la imagen redimensionada
    canvas.config(height=img_height_resized)

    resultado = predecir_imagen(file_path)

    actualizar_imagen_zoom()

    resultado_text.config(
        text=f"Predicción: {'Normal' if resultado > 0.5 else 'Neumonía'}")

    # Actualizar gráfico
    cargar_graficoMedicion(resultado)


# Función para manejar el zoom con la rueda del ratón
def hacer_zoom(event):
    global zoom_factor
    new_zoom_factor = zoom_factor * (1.01 if event.delta > 0 else 0.99)

    if 1.0 <= new_zoom_factor <= 10.0:
        zoom_factor = new_zoom_factor
        actualizar_imagen_zoom()


# Función gráfica de medición
def cargar_graficoMedicion(prediccion):
    ax.clear()
    ax.barh(0, prediccion, height=0.5,
            color='green' if prediccion > 0.5 else 'red')
    ax.set_xlim(0, 1)
    ax.set_yticks([0])
    ax.set_xticks([0, 0.5, 1])
    ax.set_xlabel('Predicción')
    probabilidad = prediccion if prediccion > 0.5 else 1 - prediccion
    ax.set_title(f"Probabilidad: {probabilidad:.2%}", fontsize=14)
    fig.canvas.draw()


# Interfaz gráfica
root = tk.Tk()
root.title("Detector de Neumonía")
root.geometry("460x850")
root.configure(bg='#f0f0f0')

# Hacer que la ventana sea de tamaño fijo
root.resizable(False, False)

# Canvas para la imagen con scrollbars
frame = tk.Frame(root, bg='#f0f0f0', bd=0, relief=tk.RAISED)
frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

canvas = Canvas(frame, width=display_width, height=display_height, bg='white')
canvas.grid(row=0, column=0, sticky='nsew')

hbar = Scrollbar(frame, orient=HORIZONTAL, command=canvas.xview)
hbar.grid(row=1, column=0, sticky='ew')

vbar = Scrollbar(frame, orient=VERTICAL, command=canvas.yview)
vbar.grid(row=0, column=1, sticky='ns')

canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)

# Enlazar eventos de zoom y desplazamiento
canvas.bind("<MouseWheel>", hacer_zoom)

# Botón para cargar imagen
btn = tk.Button(root, text="Cargar Imagen",
                command=cargar_imagen, font=("Arial", 12),
                bg='#007bff', fg='white', relief=tk.RAISED)
btn.pack(pady=10)

# Texto de resultado
resultado_text = tk.Label(root, text="", font=("Arial", 16), bg='#f0f0f0')
resultado_text.pack(pady=20)

# Gráfico
fig, ax = plt.subplots(figsize=(6, 3), facecolor='#f0f0f0')
canvas_graph = FigureCanvasTkAgg(fig, master=root)
canvas_graph.get_tk_widget().pack(fill=tk.BOTH, pady=20, expand=False)


# Configurar altura fija para el gráfico
def ajustar_altura():
    canvas_graph.get_tk_widget().config(height=200)


# Ajustar altura después de que se haya cargado el resto de la interfaz
root.after(100, ajustar_altura)

root.mainloop()
