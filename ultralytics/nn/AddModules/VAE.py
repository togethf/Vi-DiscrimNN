import torch
import torch.nn as nn
import torch.nn.functional as F
class VAELatent(nn.Module):
    def __init__(self, c1, c2=None, latent_dim=128):  
        super().__init__()
        c2 = c2 or c1  # 默认输出通道等于输入通道

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(c1, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_var = nn.Linear(256, latent_dim)
        self.fc_out = nn.Linear(latent_dim, c2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x).view(x.size(0), -1)
        mu, logvar = self.fc_mu(h), self.fc_var(h)
        z = self.reparameterize(mu, logvar)
        out = self.fc_out(z).unsqueeze(-1).unsqueeze(-1)
        return x + out  # 融合方式: residual




class SEBlock(nn.Module):
    """通道注意力模块 (Squeeze-and-Excitation)"""
    def __init__(self, c, r=16):
        super().__init__()
        self.fc1 = nn.Linear(c, c // r)
        self.fc2 = nn.Linear(c // r, c)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = x.mean((2, 3))  # GAP
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        return x * y.view(b, c, 1, 1)


class VAELatentSE(nn.Module):
    def __init__(self, c1, c2=None, latent_dim=128, dropout=0.2):
        super().__init__()
        c2 = c2 or c1

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(c1, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_var = nn.Linear(256, latent_dim)

        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(latent_dim, c2)

        self.se = SEBlock(c2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x).view(x.size(0), -1)
        mu, logvar = self.fc_mu(h), self.fc_var(h)
        z = self.reparameterize(mu, logvar)
        z = self.dropout(z)
        out = self.fc_out(z).unsqueeze(-1).unsqueeze(-1)

        # 残差 + 注意力增强
        out = self.se(x + out)
        return out

    def kl_loss(self, mu, logvar):
        """KL散度loss, 可加到总loss里"""
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
