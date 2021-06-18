#Name: Jeff Stokes
#Date: 12/07/2020
#Final - HPC and Embedded Systems
#Instructor: Gil Gallegos, gil.gallegos@gmail.com
#TA: Christopher Torres, ctorre25@live.nmhu.edu

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import time
from PIL import Image
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from SMD_N_body_system import test_system
from Random_Forest_All_Cores import random_forest
from SIR_N_scale_model import test_model

def load_data(path, header):
    # Load the CSV file into a panda dataframe
    process_data = pd.read_csv(path, header=header)
    return process_data  
   
    ###################################### PROCESSES ##########################################
def Process_1(samples,features,trees,jobs):
    return random_forest(samples,features,trees,jobs)

def Process_2(nodes):
    return test_system(nodes)

def Process_3(compartments):
    return test_model(compartments)

def mean_process_1(N):
    print("Finding mean of process 1...")
    sum_P = 0
    for i in range(N):
        result = Process_1(100,20,1000,1)
        sum_P += result
    mean_value = sum_P/N
    return(mean_value)

def mean_process_2(N):
    print("Finding mean of process 2...")
    sum_P = 0
    for i in range(N):
        result = Process_2(110)
        sum_P += result
    mean_value = sum_P/N
    return(mean_value)

def mean_process_3(N):
    print("Finding mean of process 3...\n")
    sum_P = 0
    for i in range(N):
        result = Process_3(2)
        sum_P += result
    mean_value = sum_P/N
    return(mean_value)

    ###################################### PROCESSES #########################################

if __name__ == "__main__":
   
    N_experiments = 10
    N_trials = 10
    mean_value_P1 = np.zeros([N_trials, N_experiments])
    mean_value_P2 = np.zeros([N_trials, N_experiments])
    mean_value_P3 = np.zeros([N_trials, N_experiments])
    for j in range(N_experiments):
        print("Begin experiment "+str(j+1)+"...")  
        for i in range(N_trials):
            print("Trial "+str(i+1)+": ")
            N = 5
            mean_value_P1[i][j] = mean_process_1(N)
            mean_value_P2[i][j] = mean_process_2(N)
            mean_value_P3[i][j] = mean_process_3(N)


    ###################################### Create Plots ########################################
    #flatten mean value arrays into 1xM vectors
    mean_P1_flat = mean_value_P1.flatten()
    mean_P2_flat = mean_value_P2.flatten()
    mean_P3_flat = mean_value_P3.flatten()

    print("Making scatterplots...")
    #Plot the arrays to an image and save to directory
    fig, ax1 = plt.subplots(1,3)
    fig.suptitle('Mean Times for Processes 1 to 3')
    fig.text(0.5, 0.04, 'Trial numbers 1-100', ha='center', va='center')
    fig.text(0.05, 0.5, "Time in seconds", ha='center', va='center', rotation=90)
    ax1[0].plot(mean_P1_flat)
    ax1[1].plot(mean_P2_flat)
    ax1[2].plot(mean_P3_flat)
    plt.savefig('scatterplot/mean_process_comparisons.png')
    plt.close()

    ####################################### Heat Map from Data ########################################
    print("Creating heatmaps...")
    max_mean_value_P1 = np.max(mean_value_P1)
    max_mean_value_P2 = np.max(mean_value_P2)
    max_mean_value_P3 = np.max(mean_value_P3)


    norm_P1 = mean_value_P1 / max_mean_value_P1
    norm_P2 = mean_value_P2 / max_mean_value_P2
    norm_P3 = mean_value_P3 / max_mean_value_P3

    scaled_P1 = 255*norm_P1
    scaled_P2 = 255*norm_P2
    scaled_P3 = 255*norm_P3


    type_P1 = np.uint8(scaled_P1)
    type_P2 = np.uint8(scaled_P2)
    type_P3 = np.uint8(scaled_P3)

    shape_P1 = type_P1.reshape([N_trials,N_experiments])
    shape_P2 = type_P2.reshape([N_trials,N_experiments])
    shape_P3 = type_P3.reshape([N_trials,N_experiments])


    img_P1 = Image.fromarray(shape_P1,'L')
    img_P2 = Image.fromarray(shape_P2,'L')
    img_P3 = Image.fromarray(shape_P3,'L')

    #Save grayscale subplots to directory
    fig, axs = plt.subplots(3)
    axs[0].imshow(img_P1)
    axs[1].imshow(img_P2)
    axs[2].imshow(img_P3)
    plt.suptitle('Heatmap comparisons 1-3')
    plt.savefig('images/heatmap_comparisons.png')
    plt.close()


    # Part C: create RGB image and save to file
    shape_RGB = np.dstack((img_P1,img_P2,img_P3))
    img_P_RGB = Image.fromarray(shape_RGB, 'RGB')
    img_P_RGB.save('images/image_RGB.png')

    ################################## Standard Deviation and Plotting ################################
    print("Finding standard deviation...")
    #Find standard deviation for each process value, using flattened arrays
    stdev_P1_mean = np.std(mean_P1_flat, axis=0)
    stdev_P2_mean = np.std(mean_P2_flat, axis=0)
    stdev_P3_mean = np.std(mean_P3_flat, axis=0)

    #create array containing mean values of mean_P1, mean_P2, and mean_P3 after calculating them
    dbl_mean_P1 = np.mean(mean_P1_flat)
    dbl_mean_P2 = np.mean(mean_P2_flat)
    dbl_mean_P3 = np.mean(mean_P3_flat)

    dbl_mean_P1 = np.full((N_trials*N_experiments), dbl_mean_P1)
    dbl_mean_P2 = np.full((N_trials*N_experiments), dbl_mean_P2)
    dbl_mean_P3 = np.full((N_trials*N_experiments), dbl_mean_P3)

    x_label_range = np.linspace(0,99, num=100)

    #Plot all images of standard deviation and mean comparisons for all processes
    
    fig, ax = plt.subplots(1,2)
    fig.suptitle('Standard Deviation and mean comparison for Process 1')
    ax[0].errorbar(x_label_range, mean_P1_flat, stdev_P1_mean, linestyle='None', marker='.')
    ax[1].plot(dbl_mean_P1)
    ax[1].plot(mean_P1_flat)
    plt.savefig('standard_dev_vs_mean/std_and_mean_P1.png')
    plt.close()

    fig, ax = plt.subplots(1,2)
    fig.suptitle('Standard Deviation and mean comparison for Process 2')
    ax[0].errorbar(x_label_range, mean_P2_flat, stdev_P2_mean, linestyle='None', marker='.')
    ax[1].plot(dbl_mean_P2)
    ax[1].plot(mean_P2_flat)
    plt.savefig('standard_dev_vs_mean/std_and_mean_P2.png')
    plt.close()

    fig, ax = plt.subplots(1,2)
    fig.suptitle('Standard Deviation and mean comparison for Process 3')
    ax[0].errorbar(x_label_range, mean_P3_flat, stdev_P3_mean, linestyle='None', marker='.')
    ax[1].plot(dbl_mean_P3)
    ax[1].plot(mean_P3_flat)
    plt.savefig('standard_dev_vs_mean/std_and_mean_P3.png')
    plt.close()