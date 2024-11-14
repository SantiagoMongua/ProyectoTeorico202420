import torch
from torch import nn
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from scipy.ndimage import gaussian_filter1d
device = "cuda" if torch.cuda.is_available() else "cpu"
#print(device)

n_spins = 10 
J = 1.
tau = 500
T = 30
N = 20000

df_spins = pd.read_csv("Trajectories_{}_tau_{}_nspins_{}_T_{}_J_{}.csv".format(N,tau,n_spins,T,J))
array = df_spins.values
trajectories = array.reshape(N, tau * n_spins)
        
print(trajectories.shape) #12000 100 10

df_labels = pd.read_csv("Works_labels_{}_tau_{}_nspins_{}_T_{}_J_{}.csv".format(N,tau,n_spins,T,J))
y = df_labels["Label"].values
works_train = df_labels["Works"].values

X = torch.from_numpy(trajectories).type(torch.float).view(N, tau*n_spins)
y = torch.from_numpy(y).type(torch.float)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)
X = X.to(device)

X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc

class LogisticRegression(nn.Module):
    def __init__(self,input_Dim):
        super().__init__()
        self.input_Dim = input_Dim
        self.layer_linear= nn.Linear(in_features=input_Dim,out_features=1)
        self.layer_norm = nn.LayerNorm(self.input_Dim)
        
    def forward(self, x):
        #x=self.layer_norm(x)
        x = self.layer_linear(x)
        x = x / (self.input_Dim*0.03)
        return x

model = LogisticRegression(tau*n_spins).to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01, weight_decay=0.0001)

epochs = 5000
epoch_list = np.zeros(epochs)
train_losses = np.zeros(epochs)
train_accs = np.zeros(epochs)
test_losses = np.zeros(epochs)
test_accs = np.zeros(epochs)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

for epoch in range(epochs):
    
    model.train()
    
    #y_logits = model(X_train) #Forward pass
    #print(y_logits.shape)
    y_logits = model(X_train).squeeze()
    #print(y_logits.shape)
    y_pred = torch.round(torch.sigmoid(y_logits))
    loss = loss_fn(y_logits,y_train) #Loss calculation
    acc = accuracy_fn(y_true=y_train,y_pred=y_pred)
    optimizer.zero_grad() #Optimizer to zero grad
    loss.backward()
    optimizer.step() 
    model.eval()
    
    with torch.inference_mode():
            test_logits = model(X_test).squeeze()
            test_pred = torch.round(torch.sigmoid(test_logits))
        
            test_loss = loss_fn(test_logits, y_test)
            test_losses[epoch] = test_loss
            test_acc = accuracy_fn(y_true=y_test,y_pred=test_pred)
            test_accs[epoch] = test_acc
            
    epoch_list[epoch] = epoch
    train_losses[epoch] = loss
    train_accs[epoch] = acc    
        
    if epoch%100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

#Plot model precision on training
plt.figure(figsize=(8,8))
plt.plot(epoch_list,train_accs,label="Pérdida en entrenamiento")
plt.plot(epoch_list,test_accs,label="Pérdida en prueba")
plt.xlabel("Épocas",fontsize=18)
plt.ylabel("Precisión (%)",fontsize=18)
plt.title("Precisión de predicción en prueba y en entrenamiento del modelo".format(0.005),fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.savefig("Precision_Modelo.png")
       
P = 1000

df_pruebas = pd.read_csv("Trajectories_{}_tau_{}_nspins_{}_T_{}_J_{}.csv".format(P,tau,n_spins,T,J))
array = df_pruebas.values
pruebas = array.reshape(P, tau * n_spins)

df_pruebas_labels = pd.read_csv("Works_labels_{}_tau_{}_nspins_{}_T_{}_J_{}.csv".format(P,tau,n_spins,T,J))
y_pruebas = df_pruebas_labels["Label"].values
X_pruebas = torch.from_numpy(pruebas).type(torch.float).view(P, tau*n_spins).to(device)
y_pruebas = torch.from_numpy(y_pruebas).type(torch.float).to(device)
works = df_pruebas_labels["Works"].values
 
model.eval()

with torch.inference_mode():
    test_logits = model(X_pruebas).squeeze()
    y_preds = torch.round(torch.sigmoid(test_logits))
    
print("Model accuracy: {}%".format(accuracy_fn(y_pruebas,y_preds.squeeze())))
  
beta = 1/T

def theory(x):
    sigmoide = 1+np.exp(-beta*(x))
    return 1/sigmoide

model.eval()
with torch.inference_mode():
    #test_logits = model(X_pruebas).squeeze()
    #testing = torch.sigmoid(test_logits)
    test_logits = model(X_pruebas).squeeze()
    testing = torch.sigmoid(test_logits)
    
testing = testing.cpu().numpy()

#testing = gaussian_filter1d(testing, sigma=1)
x = np.linspace(np.min(works),np.max(works),1000)
#Plotting the distribution
plt.figure(figsize=(8,8))
plt.title("$P(F|x)$ para la flecha del tiempo termodinámica",fontsize=15)
plt.scatter(works,testing,label="Predicciones del modelo",alpha = 0.35,color="lightskyblue")
plt.plot(x,theory(x),label="Predicciones teóricas",linewidth=3,color="darkblue")
plt.xlabel(r"$W$",fontsize=14)
plt.ylabel("$P(F|x)$",fontsize=14)
#plt.xlim((-10,10))
plt.legend(fontsize=12)
plt.savefig("distProb_betaGammaK1_u05_h001_t10.png")
plt.show()
