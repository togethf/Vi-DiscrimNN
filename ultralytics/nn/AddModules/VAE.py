import torch
import torch.nn as nn

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
