#Import python libraries
import numpy as np
import matplotlib.pyplot as plt
from time import process_time

def test_system(nodes):
    #Constants
    N = nodes #number of masses we have
    n = 1000 #number of iterations
    sum_T = 0 #total time for processing

    #initialize arrarys
    c = np.zeros((N,1)) #Nx1 damping coefficient vector
    k = np.zeros((N,1)) #Nx1 spring constant coefficient vector
    M = np.zeros((N,1)) #Nx1 mass vector
    x_initial = np.zeros((N,1))
    x_dot_initial = np.zeros((N,1))

    x_old = np.zeros((N,1))
    x_dot_old = np.zeros((N,1))

    f = np.zeros((N,1))

    x_dot_new = np.zeros((N,1))
    x_new = np.zeros((N,1))

    x_out = np.zeros((n,N)) #parameter space is the position of the masses, instance space is time
    x_dot_out = np.zeros((n,N))
    t_out = np.zeros((n,1)) #the size of the 

    dt = 0.001 #seconds, this is h
    dk = 0.0 #delta spring constant


    ############## Initialize Physical Parameters of System ##############
    #Beginning nodes
    c[0] = 0.1 #damping coefficient
    k[0] = 0.1 #spring constant
    M[0] = 0.5 #Kilograms, mass

    #Middle nodes
    i=1
    for i in range(N-1):
    	c[i] = 0.1
    	k[i] = 0.1+dk
    	M[i] = 0.1

    #End (Nth) nodes
    c[N-1] = 0.1
    k[N-1] = 0.1
    M[N-1] = 0.1

    #################################################################

    ##########################Initial Conditions#####################
    #Beginning Node 
    x_initial[0] = 0.0 # meters (m)
    x_dot_initial[0] = 0.0 # meters/sec (m/s)x,

    #Middle Nodes 
    i=1
    for i in range(N-1):
        x_initial[i] = 0.0 # meters (m)
        x_dot_initial[i] = 0.0 # meters/sec (m/s)x,

    #End Node
    x_initial[N-1] = -0.001 # meters (m)
    x_dot_initial[N-1] = 0.0 # meters/sec (m/s)x,
    ################################################################

    ######################## Simulation Beginning ##################
    i=0
    time = 0.0
    #Beginning Node ---> assigning initial values to old values
    x_old[0] = x_initial[0]
    x_dot_old[0] = x_dot_initial[0]

    #Middle Nodes ---> assigning initial values to old values
    i=1
    start = process_time()
    for i in range(N-1):
        x_old[i]= x_initial[i]
        x_dot_old[i] = x_dot_initial[i]

    #End Node ---> assigning initial values to old values
    x_old[N-1] = x_initial[N-1]
    x_dot_old[N-1] = x_dot_initial[N-1]
    #print(x_old,x_dot_initial)

    for i in range(n): #Beginning of simulation
        #Beginning Node ---> calculating acceleration 
        f[0] = 1/M[0]*(k[1]*(x_old[1]-x_old[0])+c[1]*(x_dot_old[1]-x_dot_old[0])-k[0]*x_old[0]-c[0]*x_dot_old[0])
        j=1
        #Middle Nodes ---> calculating acceleration
        for j in range(N-1):    
            f[j] = 1/M[j]*(-k[j]*(x_old[j]-x_old[j-1])-c[j]*(x_dot_old[j]-x_dot_old[j-1])+k[j+1]*(x_old[j+1]-x_old[j])+c[j+1]*(x_dot_old[j+1]-x_dot_old[j]))
        #End Node ---> calculating acceleration
        f[N-1] = 1/M[N-1]*(-k[N-1]*(x_old[N-1]-x_old[N-2])-c[N-1]*(x_dot_old[N-1]-x_dot_old[N-2]))
        
        #################################################
        #Calculate new velocity, x_dot_new for all nodes
        j=0
        for j in range(N):
            x_dot_new[j] = x_dot_old[j] + dt*f[j]
            
            #Calculate new position, x_new
            x_new[j] = x_old[j] + dt*x_dot_old[j]
            
            #Swapping old and new values
            x_dot_old[j] = x_dot_new[j]
            x_dot_out[i][j] = x_dot_old[j]
            x_old[j] = x_new[j]
            x_out[i][j] = x_old[j] #create the position array for all masses
        ################################################
        end = process_time()
        result = end-start

    return result