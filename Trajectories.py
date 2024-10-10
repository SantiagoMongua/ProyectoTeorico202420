import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.integrate
import time
from scipy.constants import Boltzmann

n_spins = 10 
J = 1.
B0 = 1.
tau = 100
T = 10
k_B = Boltzmann
#spins = np.random.choice([-1,1], size = n_spins)

def B(B0,t,tau):
    return B0 * np.cos((np.pi * t / tau)) 

def H(spins,B0,t,tau,J):
    interaction = - J * np.sum(spins[:-1] * spins[1:])
    field = B(B0,t,tau)
    magnetic = - field * np.sum(spins)
    return interaction + magnetic
    
def metropolis(spins, B0, t, tau, J, T, k_B):
    i = np.random.randint(0,len(spins))
    Hi = H(spins, B0, t, tau, J)
    spins[i] *= -1
    Hf = H(spins, B0, t, tau, J)
    delta_H = Hf - Hi
    if delta_H > 0:
        if np.random.rand() > np.exp(-delta_H / (k_B * T)):
            spins[i] *= -1
    
    return spins

def dBdt(B0, t, tau):
    return - B0 * (np.pi / tau) * np.sin((np.pi * t / tau))

def dWFdt(B0, t, tau, spins):   
    return - dBdt(B0, t, tau) * np.sum(spins)
    
def dWBdt(B0, t, tau, spins):
    return - dBdt(B0, (tau-t), tau) * np.sum(spins)

def trajectory(n_spins, tau, J, B0, T, k_B, forward = True):
    trajectory = np.zeros((tau, n_spins))
    spins = np.random.choice([-1, 1], size=n_spins)
    
    for t in range(tau):
        if forward:
            spins = metropolis(spins, B0, t, tau, J, T, k_B)
            trajectory[t, :] = spins  
        else:
            tr = tau - t - 1
            spins = metropolis(spins, B0, tr, tau, J, T, k_B)
            trajectory[tr, :] = spins      
    
    return trajectory

def Data(N, n_spins, B0, tau, J, T, k_B): #tau time steps
    trajectories = np.zeros((N, tau, n_spins))
    labels = []
    works = []
    steps = np.linspace(0, tau, tau)
    print("Algorithm running for {} trajectories.".format(N))
    start = time.time() #Initial running time
    for i in range(N):
        rand = np.random.random()
        if rand > 0.5:
            #Forward trajectory
            result = trajectory(n_spins, tau, J, B0, T, k_B)
            trajectories[i] = result
            work_forward = scipy.integrate.simpson([dWFdt(B0,t,tau,result) for t in range(tau)],x=steps)
            works.append(work_forward)
            labels.append(1)
            
        else:
            #Backward trajectory
            result = trajectory(n_spins, tau, J, B0, T, k_B, forward=False)
            trajectories[i] = result
            labels.append(0)
            work_backward = scipy.integrate.simpson([dWBdt(B0,t,tau,result) for t in range(tau)],x=steps)
            works.append(work_backward)
    
    data = {}

    for i in range(len(trajectories)):
        data["Label_{}".format(i)] = labels[i] 
        data["W_{}".format(i)] = works[i]
        for spin in range(len(trajectories[0][0])):
            data["Spin_{}_{}".format(i,spin)] = trajectories[i][:,spin]
            
    dataframe = pd.DataFrame(data)
    dataframe.to_csv("datosModelo.csv",index = False)
    
    end = time.time()
    print("Generated {} trajectories in {:.2f} seconds.".format(N,end-start))
    return trajectories, labels, works

Data(12000,n_spins, B0, tau, J, T, k_B)