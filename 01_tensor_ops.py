import torch
import numpy as np
x = torch.arange(1.,10.)
#Reshaping - input tensor to a defined shape
x.reshaped = x.reshape(1,9) # Must have the same number of elements
#Change the view - similar as reshaping
z = x.view(1,9) # changes in z affects x because they share the same memory
#Stacking - combine multiple tensors (concatenate) in top (v) or side (h)
x_stacked = torch.stack([x,x,x],dim = 0) #uno encima del otro also vstack
x_stacked = torch.stack([x,x,x],dim = 1) #como vectores una al lado del otro de manera vertical hstack
#Squeeze - removes all 1 dimensions from a tensor
y = torch.Tensor([[1,4,3,5,6,9]])
print(y.shape)
y = y.squeeze()
print(y.shape)
#Unsqueeze - add a 1 dimension to a target tensor
y_unsqueezed = y.unsqueeze(dim=0)
print(y_unsqueezed.shape)
#Permute - Return a view of the input with swapped dimensions, reorganises the shape
#If I modify one value, it modifies x_or and x_permuted
x_or = torch.rand(size=(224,224,3))
x_permuted = x_or.permute(2,0,1) #shift axis 0->1, 1->2, 2->0

### INDEXING
#Similar as it's done in python
x_new = torch.arange(1,10).reshape(1,3,3)
#: to select all of the target dimension, you obtain the list square bracket
print(x_new[:,:,1])
### Numpy array to tensor
array = np.arange(1.0,8.0)
tensor = torch.from_numpy(array) #Converts array to tensor reflects float64
array = array + 1 #Only the array doesn't change the tensor
tens = torch.ones(7)
nptensor = tensor.numpy()
