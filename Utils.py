# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 21:46:58 2023

@author: Luis Angel López Atencio
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
from celluloid import Camera
#import trimesh
np.set_printoptions(precision=4, suppress=True, formatter={'float': '{:0.3e}'.format})
import pyvista as pv
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
#import torch.nn.functional as F
import torch.optim as optim
# import trimesh
# import time


from matplotlib.text import Annotation
class Annotation3D(Annotation):

    def __init__(self, text, xyz, *args, **kwargs):
        super().__init__(text, xy=(0, 0), *args, **kwargs)
        self._xyz = xyz

    def draw(self, renderer):
        x2, y2, z2 = proj_transform(*self._xyz, self.axes.M)
        self.xy = (x2, y2)
        super().draw(renderer)
        
def _annotate3D(ax, text, xyz, *args, **kwargs):
    '''Add anotation `text` to an `Axes3d` instance.'''

    annotation = Annotation3D(text, xyz, *args, **kwargs)
    ax.add_artist(annotation)

setattr(Axes3D, 'annotate3D', _annotate3D)

#Funciones de herramientas matematicas
def MTHRotx(angulo):
    #Matriz de transformación homogenea, se usa la conversion de grados a radianes.
    return np.array([[1, 0, 0, 0],
                     [0, np.cos(angulo), -np.sin(angulo), 0],
                     [0, np.sin(angulo), np.cos(angulo), 0],
                     [0, 0, 0, 1]])

def MTHRoty(angulo):
    #Matriz de transformación homogenea, se usa la conversion de grados a radianes.
    return np.array([[np.cos(angulo), 0, np.sin(angulo), 0],
                     [0, 1, 0, 0],
                     [-np.sin(angulo), 0, np.cos(angulo), 0],
                     [0, 0, 0, 1]])

def MTHRotz(angulo):
    #Matriz de transformación homogenea, se usa la conversion de grados a radianes.
    return np.array([[np.cos(angulo), -np.sin(angulo), 0, 0],
                     [np.sin(angulo), np.cos(angulo), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

def MTHtrasx(distancia):
    # Crear una matriz de transformación homogénea identidad 4x4
    MTH = np.eye(4)
    # Asignar la distancia a la posición en el eje x (fila 1, columna 4)
    MTH[0, 3] = distancia
    return MTH

def MTHtrasy(distancia):
    # Crear una matriz de transformación homogénea identidad 4x4
    MTH = np.eye(4)
    # Asignar la distancia a la posición en el eje x (fila 1, columna 4)
    MTH[1, 3] = distancia
    return MTH

def MTHtrasz(distancia):
    # Crear una matriz de transformación homogénea identidad 4x4
    MTH = np.eye(4)
    # Asignar la distancia a la posición en el eje x (fila 1, columna 4)
    MTH[2, 3] = distancia
    return MTH

def dibujar_sistema_referencia_MTH(MTH, L, subindice, plotter):
    #Esta funcion grafica un sistema de referencia a partir de una matriz de transformacion homogenea.
    
    #Puntos finales de los ejes x, y, z
    pfx = MTH @ np.array([[L], [0], [0], [1]])
    pfy = MTH @ np.array([[0], [L], [0], [1]])
    pfz = MTH @ np.array([[0], [0], [L], [1]])
    
    #Origen
    origen = np.array([MTH[0, 3], MTH[1, 3], MTH[2, 3]])
    
    #Crear figura y objeto de eje si no se proporciona uno
    if plotter is None:
        plotter = pv.Plotter()
        plotter.show_grid()
        

    #Dibujar los ejes
    
    Ejex = pv.Line((origen[0], origen[1], origen[2]), (pfx[0, 0], pfx[1, 0], pfx[2, 0]))
    Ejey = pv.Line((origen[0], origen[1], origen[2]), (pfy[0, 0], pfy[1, 0], pfy[2, 0]))
    Ejez = pv.Line((origen[0], origen[1], origen[2]), (pfz[0, 0], pfz[1, 0], pfz[2, 0]))
   

    axx = plotter.add_mesh(Ejex, color='red', line_width=10)
    axy = plotter.add_mesh(Ejey, color='green', line_width=10)
    axz = plotter.add_mesh(Ejez, color='blue', line_width=10)
    
    
    #Agregar texto al final de los ejes
    subindicex = 'x'+subindice
    subindicey = 'y'+subindice
    subindicez = 'z'+subindice
    
    points = np.zeros((3,3))
    points[0,:] = pfx[:3, -1].flatten()
    points[1,:] = pfy[:3, -1].flatten()
    points[2,:] = pfz[:3, -1].flatten()
    labels = [subindicex, subindicey, subindicez]
    txt = plotter.add_point_labels(points, labels)
    
    return axx, axy, axz, txt

def dibujar_sistema_referencia_MTH_mpl(MTH, L, subindice, ax=None):
    #Esta funcion grafica un sistema de referencia a partir de una matriz de transformacion homogenea.
    
    #Puntos finales de los ejes x, y, z
    pfx = MTH @ np.array([[L], [0], [0], [1]])
    pfy = MTH @ np.array([[0], [L], [0], [1]])
    pfz = MTH @ np.array([[0], [0], [L], [1]])
    
    #Origen
    origen = np.array([MTH[0, 3], MTH[1, 3], MTH[2, 3]])
    
    #Crear figura y objeto de eje si no se proporciona uno
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        

    #Dibujar los ejes
    #ax.arrow3D(origen[0], origen[1], origen[2], pfx[0, 0], pfx[1, 0], pfx[2, 0], mutation_scale=100, arrowstyle="-", linestyle='solid', color='red')
    #ax.arrow3D(origen[0], origen[1], origen[2], pfy[0, 0], pfy[1, 0], pfy[2, 0], mutation_scale=100, arrowstyle="-", linestyle='solid', color='green')
    #ax.arrow3D(origen[0], origen[1], origen[2], pfz[0, 0], pfz[1, 0], pfz[2, 0], mutation_scale=100, arrowstyle="-", linestyle='solid', color='blue')
    ax.plot([origen[0] ,pfx[0, 0]],[origen[1],pfx[1, 0]],[origen[2],pfx[2, 0]], color = 'r')
    ax.plot([origen[0] ,pfy[0, 0]],[origen[1],pfy[1, 0]],[origen[2],pfy[2, 0]], color = 'g')
    ax.plot([origen[0] ,pfz[0, 0]],[origen[1],pfz[1, 0]],[origen[2],pfz[2, 0]], color = 'b')
    
    #Agregar texto al final de los ejes
    subindicex = 'x'+subindice
    subindicey = 'y'+subindice
    subindicez = 'z'+subindice

    ax.annotate3D(subindicex, (pfx[0, 0], pfx[1, 0], pfx[2, 0]), xytext=(3, 3), textcoords='offset points')
    ax.annotate3D(subindicey, (pfy[0, 0], pfy[1, 0], pfy[2, 0]), xytext=(3, 3), textcoords='offset points')
    ax.annotate3D(subindicez, (pfz[0, 0], pfz[1, 0], pfz[2, 0]), xytext=(3, 3), textcoords='offset points')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.grid(True)
    
def Denavit_Hartenberg(theta, d, a, alfa):  #Esta funcion calcula la matriz denavit hartenberg
    DH = np.array([[np.cos(theta), -np.cos(alfa)*np.sin(theta),  np.sin(alfa)*np.sin(theta), a*np.cos(theta)],
                   [np.sin(theta),  np.cos(alfa)*np.cos(theta), -np.sin(alfa)*np.cos(theta), a*np.sin(theta)],
                   [0, np.sin(alfa), np.cos(alfa), d],
                   [0, 0, 0, 1]])
    return DH

def DK_Raplim(q): #El parametro es el valor de las variables articulares.
    # Dimensiones del Robot
    L1=135.5;
    L2=120.23;
    L3=94.6;
    L4=65;

    # Tabla de parametros D_H
    theta_DH = np.array([(q[0]+np.pi/2),      q[1],     q[2],    q[3]]);
    d_DH     = np.array([L1,    0,        0,           0]);
    a_DH     = np.array([0,         L2,       L3,              L4]);
    alpha_DH = np.array([np.pi/2,         0,       0,             0]);

    #Matrix de transformación homogenea que representa el origen, en caso que el origen del robot no sea el punto 0,0,0,
    #es decir que este anclado a la pared se debe realizar una transormación para mover el orige, en este caso basta con la
    #siguiuente linea.
    A00 = np.eye(4)
    
    #Calcular las matricces de transformacion homogenea, con DH
    A01 = Denavit_Hartenberg(theta_DH[0], d_DH[0], a_DH[0], alpha_DH[0]);
    A12 = Denavit_Hartenberg(theta_DH[1], d_DH[1], a_DH[1], alpha_DH[1]);
    A23 = Denavit_Hartenberg(theta_DH[2], d_DH[2], a_DH[2], alpha_DH[2]);
    A34 = Denavit_Hartenberg(theta_DH[3], d_DH[3], a_DH[3], alpha_DH[3]);
    
    #Calcula matrices de transformación homogenea respecto al origen.
    A02 = A01@A12;
    A03 = A02@A23;
    A04 = A03@A34;
    
    
    #Retornar todas las matrices en una matriz de 3 dimension 5X4X4
    
    # Definir la dimensión de la matriz de matrices
    # n = 4  # número de matrices

    # Crear la matriz de matrices
    A = np.zeros((5, 4, 4))  # matriz de 5x4x4

    # Llenar la matriz de matrices con las matrices de transformación homogénea
    A[0, :, :] = A00
    A[1, :, :] = A01
    A[2, :, :] = A02
    A[3, :, :] = A03
    A[4, :, :] = A04
    
    return A

def dibujar_robot_alambres(A, L, ax):
    #Esta función grafica en almabre un robot de 6gdl
    
    #Crear figura y objeto de eje si no se proporciona uno
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    for i in range(7):
        dibujar_sistema_referencia_MTH_mpl(A[i,:,:], L, str(i), ax)


    P00 = A[0,:,3][:3] # Selecciona las tres primeras filas de la ultima columna de matriz 0
    P01 = A[1,:,3][:3]
    P02 = A[2,:,3][:3]
    P03 = A[3,:,3][:3]
    P04 = A[4,:,3][:3]
    P05 = A[5,:,3][:3]
    P06 = A[6,:,3][:3]

    #Dibujar eslabones en color negro

    ax.plot([P00[0] ,P01[0]],[P00[1],P01[1]],[P00[2],P01[2]], color = 'black')
    ax.plot([P01[0] ,P02[0]],[P01[1],P02[1]],[P01[2],P02[2]], color = 'black')
    ax.plot([P02[0] ,P03[0]],[P02[1],P03[1]],[P02[2],P03[2]], color = 'black')
    ax.plot([P03[0] ,P04[0]],[P03[1],P04[1]],[P03[2],P04[2]], color = 'black')
    ax.plot([P04[0] ,P05[0]],[P04[1],P05[1]],[P04[2],P05[2]], color = 'black')
    ax.plot([P05[0] ,P06[0]],[P05[1],P06[1]],[P05[2],P06[2]], color = 'black')
    
def Animarmov(q_inicial, q_final, i):
    #q_inicial = np.array([0, 0, 0, 0])
    #q_final = np.array([np.pi, np.pi/2, np.pi/4, np.pi/8])
    #Esta funcion anima un robot sin tener en cuenta la cinematica inversa.
    fig = plt.figure(i)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(20, -70)
    ax.set_xlim(-200,200)
    ax.set_ylim(-200,200)
    ax.set_zlim(0,400)
    ax.set_box_aspect([1,1,1])

    camera = Camera(fig) #Importante  decirle a Camera que figura es la que se va a animar

    movimiento1 = np.linspace(q_inicial[0], q_final[0], num=20)  #Guardar los valores de las variables articulares
    movimiento2 = np.linspace(q_inicial[1], q_final[1], num=20)
    movimiento3 = np.linspace(q_inicial[2], q_final[2], num=20)
    movimiento4 = np.linspace(q_inicial[3], q_final[3], num=20)

    for i in np.arange(20):
        q = np.array([movimiento1[i],movimiento2[i],movimiento3[i],movimiento4[i]])
        #Calcular cinemática direccta
        A = DK_Raplim(q)
        #Dibujar el robot en alambres en función de la cinemática directa
        dibujar_robot_alambres(A, 80, ax)
        camera.snap()

    return camera

def dibujar_raplim_pv(q, plotter):

    #Calcular cinemática direccta
    #Ojo esta funcion borra toso lo que este dibujado, su uso está orientado
    #a animaciones.
    
    A = DK_Raplim(q)
    
    # Cargar las mallas desde un archivo STL
    base = pv.read('Raplim_stls/stl_eslabon0.stl')
    eslabon1 = pv.read('Raplim_stls/stl_eslabon1.stl')
    eslabon2 = pv.read('Raplim_stls/stl_eslabon2.stl')
    eslabon3 = pv.read('Raplim_stls/stl_eslabon3.stl')
    eslabon4 = pv.read('Raplim_stls/stl_eslabon4.stl')
    
    #Aplicar transformaciones utilizando la función transform de PyVista
    base.transform(MTHtrasz(-15))
    eslabon1.transform(A[1,:,:]@MTHRotx(-np.pi/2)@MTHRotz(np.pi/2))
    eslabon2.transform(A[2,:,:]@MTHtrasx(-120.23))
    eslabon3.transform(A[3,:,:]@MTHtrasx(-94.6))
    eslabon4.transform(A[4,:,:]@MTHtrasx(-65))
    
    # Visualizar las mallas con PyVista
    #plotter = pv.Plotter()
    plotter.clear()
    # Definir luces
    luz1 = pv.Light(position=(0, 0, 1), focal_point=(0, 0, 0), color='white')
    #luz2 = pv.Light(position=(0, 0, -1), focal_point=(0, 0, 0), color='white')
    
    # Agregar luces
    plotter.add_light(luz1)

    
    plotter.add_mesh(base, color="red")
    plotter.add_mesh(eslabon1, color = "yellow")
    plotter.add_mesh(eslabon2, color = "blue")
    plotter.add_mesh(eslabon3, color = "red")
    plotter.add_mesh(eslabon4, color = "orange")
    
    #Dibujar sistemas de referencia
    dibujar_sistema_referencia_MTH(A[0,:,:], 50, '0', plotter)
    dibujar_sistema_referencia_MTH(A[1,:,:], 50, '1', plotter)
    dibujar_sistema_referencia_MTH(A[2,:,:], 50, '2', plotter)
    dibujar_sistema_referencia_MTH(A[3,:,:], 50, '3', plotter)
    dibujar_sistema_referencia_MTH(A[4,:,:], 50, '4', plotter)



def IK_Raplim(A):
    #Definir la longitud de los eslabones
    L1=135.5;
    L2=120.23;
    L3=94.6;
    L4=65;

    #Definir la posicion deseada.
    pos_final = A[0:3,3] #La primera posicion es px, la segunda py, la tercera pz

    #Calcular q1
    q1 = np.arctan2(-pos_final[0],pos_final[1])
    #q1 = round(q1,4)

    #calcular el punto de la muñeca
    x4 = A[0:3,0]  #x esta dada por la matriz de rotación 0R4

    pm = pos_final - x4*L4

    #Calcular las variables r al cuadrado (r_2) y m al cuadrado (m_2)
    #Los valores a continuación son respecto al punto de la muñeca

    r_2 = np.power(pm[0],2) + np.power(pm[1],2)
    m_2 = r_2 + np.power((pm[2]-L1),2)

    # Calcular el coseno de q3

    cos_q3 = (m_2 - np.power(L2,2) - np.power(L3,2))/(2*L2*L3)

    #Obligar a coseno de q3 que no se salga del rango
    cos_q3 = np.clip(cos_q3, -1, 1)

    #definir la variable codo

    codo = -1

    # Calcular el seno de q3
    sen_q3 = codo*np.sqrt(1-np.power(cos_q3,2))

    #Calcular q3

    q3 = np.arctan2(sen_q3,cos_q3)
    #q3 = round(q3,4)

    #Calcular q2

    beta = np.arctan2(pm[2]-L1, np.sqrt(r_2))
    alfa = np.arctan2(L3*np.sin(q3),L2+L3*np.cos(q3))

    q2 = beta - alfa

    q = np.array([q1, q2, q3, 0])

    #Calcular la matriz A03, para extraer x3, y3
    A03 = DK_Raplim(q)[3,:,:]
    x3 = A03[0:3,0]  #x esta dada por la matriz de rotación 0R3
    y3 = A03[0:3,1]  #y esta dada por la matriz de rotación 0R3

    #Calcular el coseno de q4
    cos_q4 = np.dot(x4,x3)
    #Calcular el seno de q4
    sen_q4 = np.dot(x4,y3)

    #calcular q4
    q4 = np.arctan2(sen_q4,cos_q4)
    q = np.array([q1, q2, q3, q4])
    return q


def DK_general(q, theta_DH, d_DH, a_DH, alpha_DH): 
    n = len(theta_DH)  # número de articulaciones
    theta_DH = [(q[0]+theta_DH[0]), (q[1]+theta_DH[1]), (q[2]+theta_DH[2]),
                 (q[3]+theta_DH[3]),(q[4]+theta_DH[4]),(q[5]+theta_DH[5])]
    # Calcular las matrices de transformación homogénea para cada articulación
    A = np.zeros((n + 1, 4, 4))  # matriz de n+1 x 4 x 4

    # Calcular la matriz de transformación homogénea para cada articulación
    for i in range(n):
        A[i+1, :, :] = Denavit_Hartenberg(theta_DH[i], d_DH[i], a_DH[i], alpha_DH[i])

    # Calcular las matrices de transformación homogénea respecto al origen
    A[0, :, :] = np.eye(4)  # matriz identidad de 4x4

    for i in range(1, n + 1):
        A[i, :, :] = A[i-1, :, :] @ A[i, :, :]

    # Retornar todas las matrices de transformación homogénea
    return A

def procesar_data(data_array):
    scaler_in = MinMaxScaler(feature_range=(-1,1))
    scaler_out = MinMaxScaler(feature_range=(-1,1))
    inputs_scaled = scaler_in.fit_transform(data_array[:,0:12])
    outputs_scaled = scaler_out.fit_transform(data_array[:,12:18])

    # Convertir los datos a tensores de PyTorch
    inputs_tensor = torch.from_numpy(inputs_scaled)
    outputs_tensor = torch.from_numpy(outputs_scaled)

    # Especificar el tipo de datos del tensor
    inputs_tensor = inputs_tensor.to(dtype=torch.float32)
    outputs_tensor = outputs_tensor.to(dtype=torch.float32)

    #Reshape las entradas a una matrix batch, 4X3
    Inputs_CNN2D_t = inputs_tensor.reshape(inputs_tensor.size(0), 4, 3)
    Inputs_CNN2D_t = torch.unsqueeze(Inputs_CNN2D_t, 1) #Agregar una dimension

    return Inputs_CNN2D_t, outputs_tensor

def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model = model.to(device)
    model.train()
    mae = nn.L1Loss(reduction='mean')
    train_loss, train_mae = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_mae += mae(pred, y).item()

        if batch % 100 == 0:
            current = batch * len(X)
            print(f"mse loss: {loss.item():>7f} mae loss: {mae(pred, y).item():>7f} [{current:>5d}/{size:>5d}] in batch: {batch}") 

    train_loss /= len(dataloader)  #Calcular loss promedio
    train_mae /= len(dataloader)   #Calcular maae promedio

    return train_loss, train_mae



def test_loop(dataloader, model, loss_fn, device):
    model = model.to(device)
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, test_mae = 0, 0
    mae = nn.L1Loss(reduction='mean')
    
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)         
            pred = model(X)
            test_mae += mae(pred, y).item()
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    test_mae /= num_batches
    print(f"Avg mse loss: {test_loss:>8f}\nAvg mae loss: {test_mae:>8f}")

    return test_loss, test_mae

