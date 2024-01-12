import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(in_features=10, out_features=5)

    def forward(self, x):
        return self.linear(x)


# Create an instance of the model
model = SimpleModel()

# Input tensor with 10 features
input_data = torch.randn(1, 10)
# print(input_data)
# Forward pass through the model
output = model(input_data)

# print(output)

x = torch.randn(28, 28)
# print(x)
x = x.flatten()
y = nn.Linear(28 * 28, 512)
print(dir(y))
print(y.parameters)
# print(y.weight)
# print(y.bias)
# print(y.weight.shape)
# print(y.bias.shape)
# z = y(x)
# print(y.weight)
# print(y.bias)
# z  # print(z)
