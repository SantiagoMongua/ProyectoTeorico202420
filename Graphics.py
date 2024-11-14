import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

n_spins = 10 
J = 1.
tau = 500
T = 30
N = 20000

data = pd.read_csv("Trajectories_{}_tau_{}_nspins_{}_T_{}_J_{}.csv".format(N,tau,n_spins,T,J))

trajectory = data.values.reshape(-1, tau, n_spins)[0]

plt.figure(figsize=(6, 8))
plt.imshow(trajectory, cmap="binary", aspect="auto", interpolation="nearest")
plt.colorbar(label="Spin State (+1 or -1)")
plt.xlabel("Spin",fontsize=18)
plt.ylabel("Time (t/τ)",fontsize=18)
plt.xticks(ticks=range(n_spins), labels=range(1, n_spins + 1), fontsize=12)
plt.yticks(ticks=[0, tau - 1], labels=["1", "0"], fontsize=12)
plt.show()
