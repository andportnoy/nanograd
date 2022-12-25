import torch

torch.set_default_tensor_type(torch.cuda.FloatTensor)

# this gave what looked like uninitialized memory at some point
print(torch.nn.functional.one_hot(torch.tensor([0]), 27))

# but this didn't
print(torch.nn.functional.one_hot(torch.tensor([0], device='cpu'), 27))
