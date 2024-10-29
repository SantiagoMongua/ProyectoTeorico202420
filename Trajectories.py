import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.integrate
import time

n_spins = 10 
J = 1.
B0 = 1.
tau = 100
T = 10

#spins = np.random.choice([-1,1], size = n_spins)

def B(B0,t,tau):
    return B0 * np.cos((np.pi * t / tau)) 

def H(spins,B0,t,tau,J):
    interaction = - J * np.sum(spins[:-1] * spins[1:])
    field = B(B0,t,tau)
    magnetic = - field * np.sum(spins)
    return interaction + magnetic
    
def metropolis(spins, B0, t, tau, J, T):
    i = np.random.randint(0,len(spins))
    #Hi = H(spins, B0, t, tau, J)
    spins[i] *= -1
    #Hf = H(spins, B0, t, tau, J)
    #delta_H = Hf - Hi
    
    interaction = 0
    if i == 0:
        interaction = - 2 * (spins[i] * spins[i+1])
    elif i == (len(spins) - 1):
        interaction = - 2 * (spins[i-1] * spins[i])
    else:
        interaction = - 2 * (spins[i-1] * spins[i]) - 2 * (spins[i] * spins[i+1])
    
    delta_H =   (J * interaction) - (2 * B(B0,t,tau) * spins[i])
    
    if delta_H > 0:
        if np.random.rand() > np.exp(-delta_H / (T)):
            spins[i] *= -1
    
    return spins

def trajectory_fn(n_spins, tau, J, B0, T, forward = True):
    trajectory = np.zeros((tau, n_spins))
    spins = np.random.choice([-1, 1], size=n_spins)
    workings = np.zeros(tau-1)
    
    for t in range(tau):
        if forward:
            spins = metropolis(spins, B0, t, tau, J, T)
            trajectory[t, :] = spins
            if t != (tau-1):
                workings[t] = - ((B(B0,t+1,tau) - B(B0,t,tau)) * np.sum(spins))
        
        else:
            tr = tau - t - 1
            spins = metropolis(spins, B0, tr, tau, J, T)
            trajectory[tr, :] = spins      
            if tr != 0:
                workings[tr] = - ((B(B0,tr-1,tau) - B(B0,tr,tau)) * np.sum(spins))
                
    return trajectory, workings

def Data(N, n_spins, B0, tau, J, T): #tau time steps
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
            trajectory, workings = trajectory_fn(n_spins, tau, J, B0, T)
            trajectories[i] = trajectory
            work_forward = np.sum(workings)
            works.append(work_forward)
            labels.append(1)
            
        else:
            #Backward trajectory
            trajectory, workings = trajectory_fn(n_spins, tau, J, B0, T, forward=False)
            trajectories[i] = trajectory
            work_backward = np.sum(workings)
            works.append(work_backward)
            labels.append(0)
            
    #csv for trajectories
    data_spins = {
    f"Spin_{spin}": np.hstack([trajectory[:, spin] for trajectory in trajectories])
    for spin in range(trajectories[0].shape[1])}
         
    dataframe = pd.DataFrame(data_spins)
    dataframe.to_csv("Trajectories_{}_tau_{}_nspins_{}_T_{}_J_{}.csv".format(N,tau,n_spins,T,J),index = False)
    
    #csv for labels and works
    for i in range(len(labels)):
        labels_works = {"Trajectory": i, "Label": labels[i], "Works": works[i]}
    
    dataframe = pd.DataFrame(data_spins)
    dataframe.to_csv("Works_labels_{}_tau_{}_nspins_{}_T_{}_J_{}.csv".format(N,tau,n_spins,T,J),index = False)
    
    end = time.time()
    print("Generated {} trajectories in {:.2f} seconds.".format(N,end-start))  
           
    """
    data = {}
    #Dividalo en 2 csv - labelswork y array
    for i in range(len(trajectories)):
        
        data["Label_{}".format(i)] = labels[i] 
        data["W_{}".format(i)] = works[i]
        for spin in range(len(trajectories[0][0])):
            data["Spin_{}_{}".format(i,spin)] = trajectories[i][:,spin]
            
    dataframe = pd.DataFrame(data)
    dataframe.to_csv("datosModelo.csv",index = False)
    
    end = time.time()
    print("Generated {} trajectories in {:.2f} seconds.".format(N,end-start))
    return trajectories, labels, works"""
    

Data(3,n_spins, B0, tau, J, T)