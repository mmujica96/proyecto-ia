import tkinter as tk
from tkinter import filedialog, Label
from PIL import Image, ImageTk
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


#cargar modelo
model = load_model('mi_modelo.h5')

#tamanio de la imagen que espera el modelo
img_size = 150

#funcion para procesar la imagen 
def predecir_imagen(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_size,img_size)) #ajustar el tamanio
    img = img/255.0 #normalizacion
    img = np.expand_dims(img, axis=0) #dimension de lote
    img = np.expand_dims(img, axis=-1) #dimension de canal


    prediccion = model.predict(img)[0][0]

    #normalizando valor para la escala de 0 a 1
    prediccion = min(max(prediccion, 0), 1) #
    return prediccion #retorna el valor de la prediccion (0 y 1)

    # if prediccion [0][0] > 0.5:
    #     return "Normal"
    # else:
    #     return "Neumonia"


#cargar imgagen a analizar
def cargar_imagen():
    file_path = filedialog.askopenfilename()
    img = Image.open(file_path)
    img.thumbnail((250,250)) #tama;o para mostrar en la interfaz
    img = ImageTk.PhotoImage(img)

    panel.configure(image=img)
    panel.image = img

    resultado = predecir_imagen(file_path)
    resultado_text.config(text=f"Predicción: {'Normal' if resultado > 0.5 else 'Neumonía'}")
    print(resultado)

    #actualizar grafico
    cargar_graficoMedicion(resultado)


#funcion grafica de medicion
def cargar_graficoMedicion(prediccion):
    ax.clear()
    ax.barh(0, prediccion, height=0.5, color='green' if prediccion >0.5 else 'red')
    ax.set_xlim(0,1)
    ax.set_yticks([0])
    ax.set_xticks([0, 0.5, 1])
    ax.set_xlabel('Predicion')

    ax.set_title(f"Valor: {prediccion:.8f}", fontsize=14)
    fig.canvas.draw()


#Interfaz Grafica
root = tk.Tk()
root.title("Detector de Neumonia")

panel = Label(root)
panel.pack(pady=20)


btn = tk.Button(root, text = " Cargar Imagen", command= cargar_imagen)
btn.pack(pady=10)

resultado_text = Label(root, text= "", font=("Arial", 16))
resultado_text.pack(pady=20)

#Grafico
fig, ax = plt.subplots(figsize=(5,2))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

root.mainloop()
