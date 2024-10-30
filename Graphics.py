import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

n_spins = 10 
J = 1.
tau = 500
T = 30
N = 12000

# Load the trajectories data from the CSV file
data = pd.read_csv("Trajectories_{}_tau_{}_nspins_{}_T_{}_J_{}.csv".format(N,tau,n_spins,T,J))

# Reshape the data to (time_steps, n_spins) based on your tau and n_spins values
trajectory_matrix = data.values.reshape(-1, tau, n_spins)[5]  # Select one trajectory for visualization

# Plotting the spin evolution heatmap
plt.figure(figsize=(6, 8))
plt.imshow(trajectory_matrix, cmap="binary", aspect="auto", interpolation="nearest")
plt.colorbar(label="Spin State (+1 or -1)")
plt.ylabel("Spin")
plt.ylabel("Time (t/τ)")
plt.xticks(ticks=range(n_spins), labels=range(1, n_spins + 1))
plt.yticks(ticks=[0, tau - 1], labels=["0", "1"])
plt.show()
