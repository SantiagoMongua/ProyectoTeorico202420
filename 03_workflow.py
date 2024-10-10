import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
#1. Get data ready in numerical representation
#Create data with linear regression formula
weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02
X = torch.arange(start,end,step).unsqueeze(dim=1)
Y = weight * X + bias #We know these values, we need to revise that machine learning
print(X[:10], Y[:10], len(X), len(Y))
#This is an easy way, other option with randomness is train_test_split from scikit
train_split = int(0.8 * len(X))
X_train , Y_train = X[:train_split], Y[:train_split]
X_test, Y_test = X[train_split:], Y[train_split:] #Ideal output
#VISUALIZE your data
def plot_predictions(train_data=X_train,train_labels=Y_train,test_data=X_test,test_labels=Y_test,predictions=None):
    plt.figure(figsize=(10,7))
    plt.scatter(train_data,train_labels,c ='b',s = 4, label='Training data')
    plt.scatter(test_data,test_labels,c = 'g',s=4,label="Testing data")
    if predictions is not None:
        plt.scatter(test_data, predictions, c='r',s=4, label ='Predictions')
    plt.legend(prop={"size":14})
#2. build model - linear regression
#Module is the base class for all neural network models
#The idea of the NN is to create random numbers in weights and bias and 
#adjusting them to the data. Ideally the same stablished in beginning
class LinearRegressionModel(nn.Module): #By convention in Capitalized Words
    def __init__(self): #For defining the properties of the dog in particular
        super().__init__()
        #Creates a tensor as a parameter,
        self.weights = nn.Parameter(torch.randn(1,requires_grad=True,dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1,requires_grad=True,dtype=torch.float))
    #Forward method to define the model's computation
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias

#Create a random seed
torch.manual_seed(42)
model_0 = LinearRegressionModel()
print(list(model_0.parameters()))
#List named parameters
model_0.state_dict()

# Making predictions inference mode is faster and preferred
with torch.inference_mode():
    y_preds = model_0(X_test)
    
with torch.no_grad():
    y_preds = model_0(X_test)

#plot_predictions(predictions=y_preds)

### 3. Fit (Train model)
"""The idea is to move from unknown to known parameters for a better representation
How poor is my model? Use a loss function, pytorch has a lot of them
Optimizer: Takes the loss and adjust the parameters, we need a training and a testing loop
Loss function L1Loss (Mean Average Error) Creates a criterion that measures the MAE"""
list(model_0.parameters())
# Setup of a loss function - There is a set of them in Pytorch
loss_fn = nn.L1Loss()
# Optimizer with torch.optim, a lot of them. SGD the most popular, Adam
optimizer = torch.optim.SGD(params = model_0.parameters(),lr= 0.01)

# Training loop
"""
0. Loop through the data
1. Forward pass
2. Calculate the loss
3. Optimizer zero grad
4. Loss backward - move backwards through the network to calculate gradients with respect to the loss
5. Optimizer step - use optimizer to adjust our parameters and improve the loss
One epoch is one loop, the steps 3,4,5 can be as you want in order
The idea is to find the minimum of the function, the function itself does this
Learning rate the most important hyperparameter, smaller lr, smaller step
"""
epochs = 200
#Track different values
epoch_count = []
train_loss_values = []
test_loss_values = []
for epoch in range(epochs):
    model_0.train() #train mode in Pytorch sets all parameters that require gradients to do it.
    y_pred = model_0(X_train) #Forward pass
    loss = loss_fn(y_pred,Y_train) #prediction first, then target
    optimizer.zero_grad() #Start fresh in each loop
    loss.backward() #backpropagation on the loss
    optimizer.step() #by default optimizer changes will acumulate, so its needed to zero it  
    model_0.eval() #turns off gradient tracking
    with torch.inference_mode(): # also torch.no_grad()
        # 1. Do the forward pass
        test_pred = model_0(X_test)
        # 2. Calculate the loss
        test_loss = loss_fn(test_pred, Y_test)
    if epoch % 20 == 0:
        epoch_count.append(epoch)
        train_loss_values.append(loss)
        test_loss_values.append(test_loss)
        print(f"Epoch: {epoch} | Test: {loss} | Test loss: {test_loss}") #Como van mis parametros con la evolucion del sistema
        print(model_0.state_dict())

train_loss_values = np.array(torch.tensor(train_loss_values).numpy())
test_loss_values = np.array(torch.tensor(test_loss_values).numpy())
print(train_loss_values)
plt.figure()
plt.plot(epoch_count,train_loss_values, label="Train loss")
plt.plot(epoch_count,test_loss_values, label="Train loss")
plt.title("Loss curves")
#plt.show()

"""
SAVING MY MODEL Using in other devices
1. torch.save()
2. torch.load()
"""
from pathlib import Path
# 1. Create models directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True,exist_ok=True)
# 2. Create model save path
MODEL_NAME = "model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
# 3. Save the model
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH)