import Utils
import tkinter as tk
import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from data_generator import generar
import matplotlib.pyplot as plt
from model import CNN2D
import pandas as pd
from torch.utils.data import DataLoader, random_split
from dataset import MyDataset
import torch
from torch import nn
#import threading

class App(tk.Frame):
    def __init__(self, parent):
        super(App, self).__init__(parent)
        self.data_array = None

        self.fig = Figure(figsize=(5, 4), dpi=100)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)  # A tk.DrawingArea.
        self.canvas.draw()

        self.ax = self.fig.add_subplot(111, projection="3d")
        #t = np.arange(0, 3, .01)
        #self.ax.plot(t, 2 * np.sin(2 * np.pi * t))

        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()

        self.label_progress = tk.Label(self, text="")
        self.label2 = tk.Label(self, text="PARAMETROS DENAVIT-HARTENVERG")
        self.label = tk.Label(self, text="Theta:")
        self.entry = tk.Entry(self, width=20)
        self.label3 = tk.Label(self, text="d :")
        self.entry2 = tk.Entry(self, width=20)
        self.label4 = tk.Label(self, text="a :")
        self.entry3 = tk.Entry(self, width=20)
        self.label5 = tk.Label(self, text="alpha :")
        self.entry4 = tk.Entry(self, width=20)
        self.button = tk.Button(master=self, text = "INICIAR", bg="gray69", command=self.Iniciar)#command=threading.Thread(target=self.Iniciar).start)
        self.button2 = tk.Button(master=self, text = "CALCULAR", bg="gray69", command=self.Calcular)#command=threading.Thread(target=self.Iniciar).start)

        # Entradas para el rango de variables articulares
        self.label_range = tk.Label(self, text="Rango de Variables Articulares")
        self.label_range_q1 = tk.Label(self, text="q1 (min, max):")
        self.entry_range_q1 = tk.Entry(self, width=10)
        self.label_range_q2 = tk.Label(self, text="q2 (min, max):")
        self.entry_range_q2 = tk.Entry(self, width=10)
        self.label_range_q3 = tk.Label(self, text="q3 (min, max):")
        self.entry_range_q3 = tk.Entry(self, width=10)
        self.label_range_q4 = tk.Label(self, text="q4 (min, max):")
        self.entry_range_q4 = tk.Entry(self, width=10)
        self.label_range_q5 = tk.Label(self, text="q5 (min, max):")
        self.entry_range_q5 = tk.Entry(self, width=10)
        self.label_range_q6 = tk.Label(self, text="q6 (min, max):")
        self.entry_range_q6 = tk.Entry(self, width=10)

        #self.entry.insert(0, "Hello, World!")
        self.label_progress.pack(side=tk.TOP, padx=10)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.label2.pack(side=tk.TOP)
        self.label.pack(side=tk.LEFT, padx=5)
        self.entry.pack(side=tk.LEFT, padx=5)
        self.label3.pack(side=tk.LEFT, padx=5)
        self.entry2.pack(side=tk.LEFT, padx=5)
        self.label4.pack(side=tk.LEFT, padx=5)
        self.entry3.pack(side=tk.LEFT, padx=5)
        self.label5.pack(side=tk.LEFT, padx=5)
        self.entry4.pack(side=tk.LEFT, padx=5)
        self.button.pack(side=tk.LEFT, padx=10)
        self.button2.pack(side=tk.LEFT)
        self.label_range_q1.pack(side=tk.LEFT, padx=5)
        self.entry_range_q1.pack(side=tk.LEFT, padx=5)
        self.label_range_q2.pack(side=tk.LEFT, padx=5)
        self.entry_range_q2.pack(side=tk.LEFT, padx=5)
        
        


        # Mostrar las entradas de rango
        self.label_range.pack()
        self.label_range_q3.pack()
        self.entry_range_q3.pack()
        self.label_range_q4.pack()
        self.entry_range_q4.pack()
        self.label_range_q5.pack()
        self.entry_range_q5.pack()
        self.label_range_q6.pack()
        self.entry_range_q6.pack()
        

    def Iniciar(self):
        global data_array
        self.label_progress.config(text="Generando espacio de trabajo del robot...")

        pi = 3.1416
        th = self.entry.get()
        d_ = self.entry2.get()
        a_ = self.entry3.get()
        alpha_ = self.entry4.get()
        q = np.array([0,0,0,0,0,0])

        theta = np.array([eval(th.split(",")[0]), eval(th.split(",")[1]), eval(th.split(",")[2]), 
                          eval(th.split(",")[3]), eval(th.split(",")[4]), eval(th.split(",")[5])],  dtype=np.float64)
        d     = np.array([eval(d_.split(",")[0]), eval(d_.split(",")[1]), eval(d_.split(",")[2]), 
                          eval(d_.split(",")[3]), eval(d_.split(",")[4]), eval(d_.split(",")[5])],  dtype=np.float64)
        a     = np.array([eval(a_.split(",")[0]), eval(a_.split(",")[1]), eval(a_.split(",")[2]), 
                          eval(a_.split(",")[3]), eval(a_.split(",")[4]), eval(a_.split(",")[5])],  dtype=np.float64)
        alpha = np.array([eval(alpha_.split(",")[0]), eval(alpha_.split(",")[1]), eval(alpha_.split(",")[2]), 
                          eval(alpha_.split(",")[3]), eval(alpha_.split(",")[4]), eval(alpha_.split(",")[5])],  dtype=np.float64)
        
        #Dibujar Robot en alambres

        A = Utils.DK_general(q, theta_DH=theta, d_DH=d, a_DH=a, alpha_DH=alpha)
        Utils.dibujar_robot_alambres(A, 80, self.ax)
        self.ax.set_aspect('equal')
        self.canvas.draw()

        #Extraer valor del rango de variables articulares

        q1_ = self.entry_range_q1.get()
        q2_ = self.entry_range_q2.get()
        q3_ = self.entry_range_q3.get()
        q4_ = self.entry_range_q4.get()
        q5_ = self.entry_range_q5.get()
        q6_ = self.entry_range_q6.get()

        q1_range = [eval(q1_.split(",")[0]), eval(q1_.split(",")[1])]
        q2_range = [eval(q2_.split(",")[0]), eval(q2_.split(",")[1])]
        q3_range = [eval(q3_.split(",")[0]), eval(q3_.split(",")[1])]
        q4_range = [eval(q4_.split(",")[0]), eval(q4_.split(",")[1])]
        q5_range = [eval(q5_.split(",")[0]), eval(q5_.split(",")[1])]
        q6_range = [eval(q6_.split(",")[0]), eval(q6_.split(",")[1])]


        #Rango de variables articulares

        joint_ranges = [q1_range,            # Rango para q1 en radianes
                        q2_range,            # Rango para q2 en radianes 
                        q3_range,            # Rango para q3 en radianes
                        q4_range,            # Rango para q4 en radianes
                        q5_range,            # Rango para q5 en radianes
                        q6_range]            # Rango para q6 en radianes

        #Generar dataset
        data_array = generar(theta=theta, d=d, a=a, alpha=alpha, joint_ranges=joint_ranges)

        #Abrir una nueva ventana
        top = tk.Toplevel()
        top.title("Espacio de trabajo del robot")

        #Graficar espacio de trabajo
        fig1, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(15,5))

        ax1.scatter(data_array[:,0], data_array[:,1], s=1)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title('Plano XY')

        ax2.scatter(data_array[:,0], data_array[:,2], s=1)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Z')
        ax2.set_title('Plano XZ')

        ax3.scatter(data_array[:,1], data_array[:,2], s=1)
        ax3.set_xlabel('Y')
        ax3.set_ylabel('Z')
        ax3.set_title('Plano YZ')


        #Crear dos canvases uno al lado del otro
        canvas_1 = FigureCanvasTkAgg(fig1, master=top)
        #Colocar los canvases uno al lado del otro
        canvas_1.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        #canvas_2.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)
        #canvas_2 = FigureCanvasTkAgg(fig2, master=top
        
        #Actualizar Canvas
        canvas_1.draw()
        self.label_progress.config(text="Espacio de trabajo generado")

        self.data_array = data_array
    
    def Calcular(self):
        inputs, outputs = Utils.procesar_data(self.data_array)
        dataset = MyDataset(inputs, outputs)  #Crear la instancia

        #Dividir los datos
        TEST_RATIO = 0.15
        VAL_RATIO = 0.15
        BATCH_SIZE = 64

        size_all = len(dataset)
        #print(f'Antes de dividir el dataset, este es su tamaño: len(dataset_CNN2D)={size_all}')

        size_test = int(size_all * TEST_RATIO)
        size_val =  int(size_all * VAL_RATIO)
        size_train = size_all - size_test - size_val

        dataset_train, dataset_test, dataset_val = random_split(dataset, [size_train, size_test, size_val])
        #print(f'Después de dividir en conjuntos de entrenamiento, prueba y validación: len(dataset_train_CNN2D)={len(dataset_train_CNN2D)}. len(dataset_test_CNN2D)={len(dataset_test_CNN2D)}. len(dataset_val_CNN2D)={len(dataset_val_CNN2D)}')

        trainloader = DataLoader(dataset_train, batch_size = BATCH_SIZE)
        testloader = DataLoader(dataset_test, batch_size = BATCH_SIZE)
        valloader = DataLoader(dataset_val, batch_size = BATCH_SIZE)

        #Entrenar el modelo
        model = CNN2D(stride=1, padding=1, kernel=2, out_channels=6 )
        #print(f"Model structure: {model6}\n\n")
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001,)

        epochs = 20
        metricas_mse_train_test = np.zeros((epochs, 3))
        metricas_mae_train_test = np.zeros((epochs, 3))
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            metricas_train = Utils.train_loop(trainloader, model, loss_fn, optimizer, "cpu")
            print(f"test {t+1}\n-------------------------------")
            metricas_test  = Utils.test_loop(testloader, model, loss_fn, "cpu")
            print(f"val {t+1}\n-------------------------------")
            metricas_val  = Utils.test_loop(valloader, model, loss_fn, "cpu")
            
            
            metricas_mse_train_test[t][0] = metricas_train[0]
            metricas_mse_train_test[t][1] = metricas_test[0]
            metricas_mse_train_test[t][2] = metricas_val[0]
            
            
            metricas_mae_train_test[t][0] = metricas_train[1]
            metricas_mae_train_test[t][1] = metricas_test[1]
            metricas_mae_train_test[t][2] = metricas_val[1]
        print("Done!")

        r2 = 1 - (metricas_mse_train_test[t][1] / torch.var(dataset_test[:][1]))
        print(r2)

        #Guardar modelo

        model_state_dict = model.state_dict()
        optimizer_state_dict = optimizer.state_dict()

        checkpoint = {
            "model_state_dict" :  model_state_dict,
            "optimizer_state_dict" : optimizer_state_dict,
            "epoch" : epochs,
            "loss" : metricas_mse_train_test[t][0]
        }

        torch.save(checkpoint, "model_checkpoint.pth")
