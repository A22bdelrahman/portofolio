import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Simulate two clients with models
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        return self.fc(x)

# Client data
data1 = torch.tensor([[1.,1.], [2.,2.]])
targets1 = torch.tensor([2., 4.])
data2 = torch.tensor([[3.,3.], [4.,4.]])
targets2 = torch.tensor([6., 8.])

model1 = SimpleModel()
model2 = SimpleModel()
crit = nn.MSELoss()
opt1 = optim.SGD(model1.parameters(), lr=0.01)
opt2 = optim.SGD(model2.parameters(), lr=0.01)

# Local train
for _ in range(3):
    opt1.zero_grad()
    out1 = model1(data1)
    loss1 = crit(out1.squeeze(), targets1)
    loss1.backward()
    opt1.step()

    opt2.zero_grad()
    out2 = model2(data2)
    loss2 = crit(out2.squeeze(), targets2)
    loss2.backward()
    opt2.step()

# Federate: average weights
global_model = SimpleModel()
for p1, p2, pg in zip(model1.parameters(), model2.parameters(), global_model.parameters()):
    pg.data = (p1.data + p2.data) / 2

# Test global
test_data = torch.tensor([[5.,5.]])
pred = global_model(test_data).item()
print(f"Federated model predicts {pred:.2f} for input [5,5] (expected ~10)")