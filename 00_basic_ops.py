import torch

scalar = torch.tensor(7)
scalar.item #Show as a Python int
vector = torch.tensor([7,7])
vector.ndim #How many vectors or square brackets
vector.shape #How many elements 
matrix = torch.tensor([[7,8],[9,10]])
print(matrix[1]) #The second row
matrix.shape #The composition of the matrix Ex. 2*2
TENSOR = torch.tensor([[[1,2,3],[3,6,9],[2,4,5]]]) #TENSOR and MATRIX in uppercase for coding
TENSOR.shape #Three numbers for each square brackets is a 1 3x3 tensor

###RANDOM TENSORS
random_tensor = torch.rand(3,4) #size 3,4 from an uniform distribution
random_image_tensor = torch.rand(size=(224,224,3)) #3 for rgb, sometimes RGB at beggining
###zeros AND ONES
zeros = torch.zeros(size=(3,4))
ones = torch.ones(size=(3,4))
## range of tensors and tensors like
one_to_ten = torch.arange(start=1,end=10,step=1) #Includes both
ten_zeroes = torch.zeros_like(one_to_ten) #In the same shape as one to ten
### Tensor datatypes - 3 big errors with Pytorch
# Not right datatype tensor.dtype
# Not right shape tensor.shape
# Not on the right device tensor.device
# Precision in computing, measure of detail 32 + precise que 16
#dtype_tensor = torch.tensor([3.0,6.0,9.0],dtype=None,device=None,requires_grad=False)
print(ten_zeroes.device)
### TENSOR OPERATIONS
tensor = torch.tensor([1,2,3])
add = tensor + 10 # Suma (o resta) elemento por elemento
mult = tensor * 10 # Multiplica (o divide) elemento por elemento
add = tensor + 10 # Suma elemento por elemento
ans = torch.matmul(tensor,tensor) # (or mm) Matrix multiplication as known dot product
# Matmul of (2,3) & (3,2) work but (3,2) & (3,2) not work
# The resulting matrix has the outer dimensions shape
tensor_A = torch.tensor([[1,2],[3,4],[5,6]])
tensor_B = torch.tensor([[7,10],[8,11],[9,12]])
# Transpose function to fix the shapes, switches the axes
transposed = tensor_B.T # (2,3) goes to (3,2)
torch.mm(tensor_A,transposed)

## AGGREGATION OPS
x = torch.arange(0,100,10)
torch.min(x) # Also x.min()
torch.argmin(x) # Also x.argmin(), position where the minimum is found
torch.argmax(x) # Also x.argmax(), position where the maximum is found
torch.max(x) # Also x.max()
torch.mean(x.type(torch.float32)) # Alsox.type().mean(), float32 to work
torch.sum(x) # Also x.sum()
