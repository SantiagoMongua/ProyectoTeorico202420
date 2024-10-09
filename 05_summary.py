import torch
from torch import nn
import matplotlib.pyplot as plt
### DATA
weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

def plot_predictions(train_data=X_train,train_labels=y_train,test_data=X_test,test_labels=y_test,predictions=None):
    plt.figure(figsize=(10,7))
    plt.scatter(train_data,train_labels,c ='b',s = 4, label='Training data')
    plt.scatter(test_data,test_labels,c = 'g',s=4,label="Testing data")
    if predictions is not None:
        plt.scatter(test_data, predictions, c='r',s=4, label ='Predictions')
    plt.legend(prop={"size":14})
    plt.show()
# Building model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1) # 1 x (in) for each y (out) 
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)
    
torch.manual_seed(42)
model_1 = LinearRegressionModel()
device = torch.device('cuda:0')
model_1.to(device)
print(model_1, model_1.state_dict())

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.01)

torch.manual_seed(42)
epochs = 200

X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

#Training model
for epoch in range(epochs):
    model_1.train()
    y_pred = model_1(X_train) #Forward pass
    loss = loss_fn(y_pred,y_train) #Loss calculation
    optimizer.zero_grad() #Optimizer to zero grad
    loss.backward() #Backpropagation
    optimizer.step() #Optimizer step
    #Testing
    model_1.eval()
    with torch.inference_mode():
        test_pred = model_1(X_test)
        test_loss = loss_fn(test_pred, y_test)
        
    if epoch%10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")
        
#Evaluating the model        
model_1.eval()
with torch.inference_mode():
    y_preds = model_1(X_test)

plot_predictions(predictions=y_preds.cpu())

#Saving model
from pathlib import Path

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True,exist_ok=True)

MODEL_NAME = "model_1.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_1.state_dict(), f=MODEL_SAVE_PATH)