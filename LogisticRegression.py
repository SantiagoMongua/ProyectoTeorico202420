import torch
from torch import nn
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

n_spins = 10 
J = 1.
tau = 500
T = 10
N = 12000

df_spins = pd.read_csv("Trajectories_{}_tau_{}_nspins_{}_T_{}_J_{}.csv".format(N,tau,n_spins,T,J))
array = df_spins.values
trajectories = array.reshape(N, tau * n_spins)
        
print(trajectories.shape) #12000 100 10

X = torch.from_numpy(trajectories).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)

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
        self.layer_1 = nn.Linear(in_features=input_Dim,out_features=1)
        
    def forward(self, x):
        return self.layer_1(x)

model = LogisticRegression(100,10).to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.1)
#Un vector lineal
epochs = 500
epoch_list = np.zeros(epochs)
train_losses = np.zeros(epochs)
train_accs = np.zeros(epochs)
test_losses = np.zeros(epochs)
test_accs = np.zeros(epochs)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

for epoch in range(epochs):
    model.train()
    y_logits = model(X_train).squeeze() #Forward pass
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