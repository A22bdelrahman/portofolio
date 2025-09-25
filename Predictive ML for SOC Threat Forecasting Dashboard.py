import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Generate synthetic time series data
t = np.linspace(0, 10, 100)
data = np.sin(t) + 0.1 * np.random.randn(100)
data = data.reshape(-1, 1, 1)  # (seq, batch, feature)

data_tensor = torch.tensor(data[:-1], dtype=torch.float32)
targets = torch.tensor(data[1:], dtype=torch.float32)

# LSTM model
class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 10, batch_first=True)
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h.squeeze(0))

model = LSTM()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train
for epoch in range(5):
    optimizer.zero_grad()
    output = model(data_tensor)
    loss = criterion(output, targets.squeeze(1))
    loss.backward()
    optimizer.step()

print(f"LSTM trained with final loss {loss.item():.4f}")