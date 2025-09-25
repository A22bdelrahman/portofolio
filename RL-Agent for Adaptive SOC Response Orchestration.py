import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Simple environment: 2 states, 2 actions
states = 2
actions = 2

# Q-Network
class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, actions)  # State as scalar for simplicity

    def forward(self, s):
        return self.fc(s)

qnet = QNet()
optimizer = optim.Adam(qnet.parameters(), lr=0.1)
criterion = nn.MSELoss()

# Mock training
for episode in range(10):
    state_value = np.random.randint(0, states)
    state = torch.tensor([[state_value]], dtype=torch.float32)
    action = np.random.randint(0, actions)
    reward = 1 if action == 0 else -1
    next_state_value = (state_value + 1) % states
    next_state = torch.tensor([[next_state_value]], dtype=torch.float32)

    q_values = qnet(state)
    next_q = qnet(next_state).max(1)[0]
    target = torch.tensor(reward) + 0.9 * next_q

    loss = criterion(q_values[0, action], target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"RL Q-Net trained with final loss {loss.item():.4f}")