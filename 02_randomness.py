import torch
#reproducibility with random seed pseudo random
ran_A = torch.rand(3,4)
ran_B = torch.rand(3,4)
seed = 42
torch.manual_seed(seed)
ran_C = torch.rand(3,4)
torch.manual_seed(seed) #This line its needed because the seed is reset
ran_D = torch.rand(3,4)
device = "cuda" if torch.cuda.is_available() else "cpu"
tensor = torch.tensor([1,2,3])
ten_gpu = tensor.to(device) #move tensor to gpu if available
ten_cpu = ten_gpu.cpu()#Back to cpu
