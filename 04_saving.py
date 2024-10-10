import torch
from torch import nn
class LinearRegressionModel(nn.Module): #By convention in Capitalized Words
    def __init__(self): #For defining the properties of the dog in particular
        super().__init__()
        #Creates a tensor as a parameter,
        self.weights = nn.Parameter(torch.randn(1,requires_grad=True,dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1,requires_grad=True,dtype=torch.float))
    #Forward method to define the model's computation
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias
    
loaded_model_0 = LinearRegressionModel()
loaded_model_0.load_state_dict(torch.load(f='models\model_0.pth'))

