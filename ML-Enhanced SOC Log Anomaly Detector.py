import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Generate synthetic log data: normal and anomalous
np.random.seed(42)
normal_data = np.random.normal(0, 1, (1000, 10))
anomalous_data = np.random.normal(5, 1, (100, 10))
data = np.vstack([normal_data, anomalous_data])
data_tensor = torch.tensor(data, dtype=torch.float32)

# Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(10, 5), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(5, 10), nn.ReLU())

    def forward(self, x):
        return self.decoder(self.encoder(x))

model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train
for epoch in range(5):
    optimizer.zero_grad()
    output = model(data_tensor)
    loss = criterion(output, data_tensor)
    loss.backward()
    optimizer.step()

# Test reconstruction error
recon = model(data_tensor)
errors = torch.mean((recon - data_tensor)**2, dim=1)
threshold = errors.mean() + 2 * errors.std()
anomalies = (errors > threshold).sum().item()

print(f"Detected {anomalies} anomalies out of 1100 samples.")