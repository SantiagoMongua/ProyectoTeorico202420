import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

"""
    This file allows to create the trajectories of a spin chain protocol, using the coupling constant in a fixed value
    and changing the magnetic field in a sinusoidal way.
    It uses the Metropolis algorithm to calculate the change on the spins in the time evolution.
    Finally, it saves the data in a csv file for training a neural network to predict the directionality of the process
"""

n_spins = 10 
J = 1.
B0 = 20.
tau = 500
T = 30

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
    spins[i] *= -1 # Flip the spin temporarily
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
            spins[i] *= -1 # Undo the flip if move is rejected
    
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
                workings[tr-1] = - ((B(B0,tr-1,tau) - B(B0,tr,tau)) * np.sum(spins))
       
    return trajectory, workings

def Data(N, n_spins, B0, tau, J, T):
    trajectories = np.zeros((N, tau, n_spins))
    labels = []
    works = []
    print("Algorithm running for {} trajectories.".format(N))
    start = time.time()
    for i in range(N):
        rand = np.random.random()
        if rand > 0.5:
            # Forward trajectory
            trajectory, workings = trajectory_fn(n_spins, tau, J, B0, T)
            trajectories[i] = trajectory
            work_forward = np.sum(workings)
            works.append(work_forward)
            labels.append(1)
            
        else:
            # Backward trajectory
            trajectory, workings = trajectory_fn(n_spins, tau, J, B0, T, forward=False)
            trajectories[i] = trajectory
            work_backward = np.sum(workings)
            works.append(work_backward)
            labels.append(0)
            
    # Save spin trajectories
    data_spins = {
        f"Spin_{spin}": np.hstack([trajectory[:, spin] for trajectory in trajectories])
        for spin in range(trajectories[0].shape[1])
    }
         
    df_spins = pd.DataFrame(data_spins)
    df_spins.to_csv("Trajectories_{}_tau_{}_nspins_{}_T_{}_J_{}.csv".format(N,tau,n_spins,T,J),index = False)
    
    # Save labels and works
    labels_works = [{"Trajectory": i, "Label": labels[i], "Works": works[i]} for i in range(N)]
    
    df_works = pd.DataFrame(labels_works)
    df_works.to_csv("Works_labels_{}_tau_{}_nspins_{}_T_{}_J_{}.csv".format(N,tau,n_spins,T,J),index = False)
    
    end = time.time()
    print("Generated {} trajectories in {:.2f} seconds.".format(N,end-start))      

# Run data generation
Data(12000,n_spins, B0, tau, J, T)