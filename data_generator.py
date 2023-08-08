import numpy as np
import matplotlib.pyplot as plt
import Utils

def generar(theta, d, a, alpha, joint_ranges):

    np.random.seed(42)  # Establecer la semilla para reproducibilidad
    n_samples = 20000  # número de muestras

    # Define ranges for each joint angle
    

    # Crear una matriz para almacenar las muestras generadas
    samples = np.zeros((n_samples, 6))

    for j in range(6):
        # Generar muestras aleatorias para posición
        samples[:, j] = np.random.uniform(joint_ranges[j][0], joint_ranges[j][1], size=n_samples)
        # Generar muestras aleatorias para orientación

    #Variable articular
    q_values = np.array(samples)
    data = []
    for i in range(n_samples):
        
        q = q_values[i]
        
        A = Utils.DK_general(q, theta_DH=theta, d_DH=d, a_DH=a, alpha_DH=alpha)
        
        x = A[6, 0, 3]
        y = A[6, 1, 3]
        z = A[6, 2, 3]
        
        nx1 = A[6, 0, 0]
        nx2 = A[6, 1, 0]
        nx3 = A[6, 2, 0]
        ox3 = A[6, 2, 1]
        ax3 = A[6, 2, 2]
        
        #Calcular angulos yaw, pitch, roll, la matriz de rotacion está dada por la multiplicación de los
        #angulos yaw, pitch, roll en ese orde
        
        yaw =   np.arctan2(nx2,nx1)
        pitch = np.arctan2(-nx3, np.sqrt(np.float_power(ox3,2) + np.float_power(ax3,2)))
        roll =  np.arctan2(ox3, ax3)

        input_position_orientation = (np.round(x,4), np.round(y,4), np.round(z,4), np.round(yaw,4), np.round(pitch,4), np.round(roll,4))
        outputjoints = (np.round(q[0],4),np.round(q[1],4),np.round(q[2],4),np.round(q[3],4),np.round(q[4],4),np.round(q[5],4))
        
        #Generar el q actual
        for sample in range(50):
            inputjoints = (np.round(q[0] + 0.3*(np.random.random()-0.5) ,4), #Generar q aleatorios cerca del q actual
                        np.round(q[1] + 0.3*(np.random.random()-0.5) ,4),
                        np.round(q[2] + 0.3*(np.random.random()-0.5) ,4),
                        np.round(q[3] + 0.3*(np.random.random()-0.5) ,4),
                        np.round(q[4] + 0.3*(np.random.random()-0.5) ,4),
                        np.round(q[5] + 0.3*(np.random.random()-0.5) ,4)
                        )
            
            
            data.append( [ input_position_orientation, inputjoints , outputjoints ] )
            
    data_array = np.array(data).reshape(len(data), 18)
#Graficar el espacio de trabajo o los puntos con el cual será entrenado el modelo

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(15,5))

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


    return data_array