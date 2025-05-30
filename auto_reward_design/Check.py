import torch

device = torch.cuda.is_available()
print(f'GPU: {device}')

torch.cuda.empty_cache()