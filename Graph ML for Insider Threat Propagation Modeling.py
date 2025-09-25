import torch
import torch.nn as nn
import numpy as np
import networkx as nx

# Similar to above, small graph
G = nx.random_tree(20)
adj = nx.to_numpy_array(G) + np.eye(20)  # Add self-loops
features = np.random.rand(20, 5)
risks = np.random.rand(20)  # Mock propagation scores

adj_t = torch.tensor(adj, dtype=torch.float32)
feat_t = torch.tensor(features, dtype=torch.float32)
risks_t = torch.tensor(risks, dtype=torch.float32)

# Simple propagation model
class PropNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 1)

    def forward(self, adj, feat):
        prop = torch.mm(adj, feat)
        return torch.sigmoid(self.fc(prop)).squeeze()

model = PropNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Train
for epoch in range(5):
    optimizer.zero_grad()
    output = model(adj_t, feat_t)
    loss = criterion(output, risks_t)
    loss.backward()
    optimizer.step()

print(f"Graph ML model trained with final loss {loss.item():.4f}")