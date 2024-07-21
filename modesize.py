import torch
import time


model = torch.load('saveModel/Chemport_noParams.pth')
model.eval()


print(f"Chemport_noParams.pth")

model_size = sum(p.numel() for p in model.parameters()) * 4 / (1024 ** 2)
print(f"Model size: {model_size:.2f} MB")

param_count = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {param_count}")
