import torch
import torch.nn as nn

# Define MSELoss
criterion = nn.MSELoss()

# Example input and target tensors
input_data = torch.randn((3, 5), requires_grad=True)
target_data = torch.randn((3, 5), requires_grad=False)

# Compute the mean squared error
loss = criterion(input_data, target_data)
print(input_data)
print(target_data)
print(loss.item())  # Access the value as a Python float
