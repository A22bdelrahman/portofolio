import torch
import torch.nn as nn
import torch.optim as optim

# Generator and Discriminator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 20)

    def forward(self, x):
        return self.fc(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(20, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(x))

G = Generator()
D = Discriminator()
criterion = nn.BCELoss()
g_opt = optim.Adam(G.parameters(), lr=0.01)
d_opt = optim.Adam(D.parameters(), lr=0.01)

# Synthetic real data
real = torch.ones(32, 20)
noise = torch.randn(32, 10)

# Train step
for _ in range(5):
    # Discriminator
    d_opt.zero_grad()
    fake = G(noise)
    d_real = D(real)
    d_fake = D(fake.detach())
    d_loss = criterion(d_real, torch.ones(32, 1)) + criterion(d_fake, torch.zeros(32, 1))
    d_loss.backward()
    d_opt.step()

    # Generator
    g_opt.zero_grad()
    d_fake = D(fake)
    g_loss = criterion(d_fake, torch.ones(32, 1))
    g_loss.backward()
    g_opt.step()

print(f"GAN trained with final G loss {g_loss.item():.4f}, D loss {d_loss.item():.4f}")