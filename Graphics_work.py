import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

n_spins = 10 
J = 1.
tau = 500
T = 50
N = 12000

data = pd.read_csv("Works_labels_{}_tau_{}_nspins_{}_T_{}_J_{}.csv".format(N,tau,n_spins,T,J))
print(data.head())
work_forward = data[data["Label"] == 1]["Works"]
work_backward = data[data["Label"] == 0]["Works"]

mean_backward = np.mean(work_backward)
mean_forward = np.mean(work_forward)

plt.figure(figsize=(6, 8))
plt.hist(work_forward, bins=30, density=True, alpha = 0.6, color='red', label="Forward")
plt.hist(work_backward, bins=30, density=True, alpha = 0.4, color='blue', label="Backward")
plt.axvline(mean_forward, color='red', linestyle='-', linewidth=1.5, label=r'$\langle \beta W \rangle_F$ = {:.2f}'.format(mean_forward))
plt.axvline(mean_backward, color='blue', linestyle='-', linewidth=1.5, label=r'$\langle \beta W \rangle_B$ = {:.2f}'.format(mean_backward))
plt.xlabel(r"$W$")
plt.ylabel(r"$\rho(W)$")
plt.legend()
plt.tight_layout()
plt.show()

