# Python program to show time by process_time() 
import numpy as np 
import matplotlib.pyplot as mp
from time import process_time

def test_model(compartments):
    #Constants
    n = 100 #Number of time iterations, dt = 0.001 s, T = n*dt = 1 s 
    N = compartments
    g = 1/3
    B = np.abs(-1/6*(np.random.random((N,N))+1/2))


    s_initial = np.zeros((N,1))
    i_initial = np.zeros((N,1))
    r_initial = np.zeros((N,1))

    for i in range(N):
        s_initial[i] = 1 #N1/Nmax N2/Nmax  Nmax = max{N2,N2}
        i_initial[i] = 1.27e-6
        r_initial[i] = 0
        
    s_old = np.zeros((N,1))
    s_new = np.zeros((N,1))

    i_old = np.zeros((N,1))
    i_new = np.zeros((N,1))

    r_old = np.zeros((N,1))
    r_new = np.zeros((N,1))

    f_s_old = np.zeros((N,1))
    f_s_new = np.zeros((N,1))
    f_i = np.zeros((N,1))
    f_r = np.zeros((N,1))

    s_out = np.zeros((n,N))
    r_out = np.zeros((n,N))
    i_out = np.zeros((n,N))
    t_out = np.zeros((n,1))

    dt = 0.001 #seconds 

    ######################## Simulation Beginning ##################
    time = 0.0
    start = process_time()
    #Beginning ---> assigning initial values to old values
    for i in range(N):
        s_old[i] = s_initial[i]
        i_old[i] = i_initial[i]
        r_old[i] = r_initial[i]
    time = 0   
    for k in range(2):
        for i in range(N):
            sum_f = 0
            for j in range(N):
                sum_f = sum_f + B[i][j]*s_old[i]*i_old[j]
                
            f_s_new[i] = -sum_f
            f_i[i] = -f_s_old[i] -g*i_old[i]
            f_r[i] = g*i_old[i]
            
            #Euler integration
            s_new[i] = s_old[i] + f_s_new[i]*dt
            i_new[i] = i_old[i] + f_i[i]*dt
            r_new[i] = r_old[i] + f_r[i]*dt
            
            #swap old and new values
            s_old[i] = s_new[i]
            i_old[i] = i_new[i]
            r_old[i] = r_new[i]
            f_s_old[i] = f_s_new[i]
            
            #output values
            s_out[k][i] = s_old[i]
            i_out[k][i] = i_old[i]
            r_out[k][i] = r_old[i]
        t_out[k]  = time   
        time = time + dt
    end = process_time()
    result = end - start
    return result