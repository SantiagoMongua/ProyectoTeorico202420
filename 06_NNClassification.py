import torch
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import requests
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

n_samples = 1000
X, y = make_circles(n_samples, noise = 0.03, random_state=42)
circles = pd.DataFrame({"X1": X[:,0], "X2": X[:,1], "label": y})
plt.scatter(x=X[:,0], y=X[:,1],c=y,cmap=plt.cm.RdYlBu)
plt.show()
#The idea is to separate the item between the two circles
#Always view the shapes of inputs and outputs, it could be with first data
X_sample = X[0]
y_sample = y[0]
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)

class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2,out_features=5) # More hidden layers mpre opportunity to learn patterns
        self.layer_2 = nn.Linear(in_features=5,out_features=1) #The in must match the out of the first one
        
    def forward(self, x):
        return self.layer_2(self.layer_1(x))

model_2 = CircleModelV0().to(device)

#This code works equal to the class that we created with nn.Module in an easiest way
 
model_2 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1)
).to(device)

with torch.inference_mode():
    untrained_preds = model_2(X_test.to(device))

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model_2.parameters(), lr=0.1)
#Model outputs are raw logits, it must be converted to probabilities by activation function
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc

X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

torch.manual_seed(42)
epochs = 100

for epoch in range(epochs):
    model_2.train()
    y_logits = model_2(X_train).squeeze() #Forward pass
    y_pred = torch.round(torch.sigmoid(y_logits))
    loss = loss_fn(y_logits,y_train) #Loss calculation
    acc = accuracy_fn(y_true=y_train,y_pred=y_pred)
    optimizer.zero_grad() #Optimizer to zero grad
    loss.backward()
    optimizer.step() 
    model_2.eval()
    with torch.inference_mode():
        test_logits = model_2(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test,y_pred=test_pred)
        
    if epoch%10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}%| Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

if Path("helper_functions.py").is_file():
    print("helper_functions.py already exists, skipping download")
else:
    print("Download helper_functions.py")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/refs/heads/main/helper_functions.py")
    with open("helper_functions.py", "wb") as f:
        f.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(model_2, X_train, y_train)
plt.subplot(1,2,2)
plt.title("Test")
plot_decision_boundary(model_2,X_test,y_test)
plt.show()

"""Improving a model
Add more layers
Add more hidden units 
Fit for longer
Changing the activation functions
Change the learning rate
"""

class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10,out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
    
    def forward(self, x):
        return self.layer_3(self.layer_2(self.layer_1(x)))
    
model_3 = CircleModelV1().to(device)
optimizer_1 = torch.optim.SGD(params=model_3.parameters(), lr=0.1)

X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
epochs = 1000

for epoch in range(epochs):
    model_3.train()
    y_logits = model_3(X_train).squeeze() #Forward pass
    y_pred = torch.round(torch.sigmoid(y_logits))
    loss = loss_fn(y_logits,y_train) #Loss calculation
    acc = accuracy_fn(y_true=y_train,y_pred=y_pred)
    optimizer.zero_grad() #Optimizer to zero grad
    loss.backward()
    optimizer_1.step() 
    model_3.eval()
    with torch.inference_mode():
        test_logits = model_3(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test,y_pred=test_pred)
        
    if epoch%100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}%| Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")
        
        
        
