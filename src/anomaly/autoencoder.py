"""
Autoencoder convolutionnel pour la detection d'anomalies.
Entraine sur des images normales ; le score d'anomalie est l'erreur de reconstruction (MSE).
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    """Encodeur convolutionnel : 4 blocs de downsampling par stride=2."""

    def __init__(self, in_channels: int = 3, base_channels: int = 32, latent_dim: int = 128):
        super().__init__()
        c = base_channels
        self.conv_blocks = nn.Sequential(
            self._block(in_channels, c),       # 64->32
            self._block(c,    c * 2),          # 32->16
            self._block(c*2,  c * 4),          # 16->8
            self._block(c*4,  c * 8),          # 8->4
        )
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(c * 8 * 4 * 4, latent_dim)

    @staticmethod
    def _block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_blocks(x)
        h = self.flatten(h)
        return self.fc_mu(h)


class VAEEncoder(nn.Module):
    """Encodeur VAE : retourne mu et logvar pour la reparametrisation."""

    def __init__(self, in_channels: int = 3, base_channels: int = 32, latent_dim: int = 128):
        super().__init__()
        c = base_channels
        self.conv_blocks = nn.Sequential(
            self._block(in_channels, c),
            self._block(c,    c * 2),
            self._block(c*2,  c * 4),
            self._block(c*4,  c * 8),
        )
        self.flatten = nn.Flatten()
        flat_dim = c * 8 * 4 * 4
        self.fc_mu     = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

    @staticmethod
    def _block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.flatten(self.conv_blocks(x))
        return self.fc_mu(h), self.fc_logvar(h)


class ConvDecoder(nn.Module):
    """Decodeur convolutionnel : 4 blocs d'upsampling par ConvTranspose2d."""

    def __init__(self, out_channels: int = 3, base_channels: int = 32, latent_dim: int = 128):
        super().__init__()
        c = base_channels
        self.fc = nn.Linear(latent_dim, c * 8 * 4 * 4)
        self.deconv_blocks = nn.Sequential(
            self._block(c * 8, c * 4),  # 4->8
            self._block(c * 4, c * 2),  # 8->16
            self._block(c * 2, c),      # 16->32
            nn.ConvTranspose2d(c, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _block(in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(z.size(0), -1, 4, 4)
        return self.deconv_blocks(h)


class ConvAutoencoder(nn.Module):
    """Autoencoder convolutionnel. Perte = MSE reconstruction."""

    def __init__(self, in_channels: int = 3, base_channels: int = 32, latent_dim: int = 128):
        super().__init__()
        self.encoder = ConvEncoder(in_channels, base_channels, latent_dim)
        self.decoder = ConvDecoder(in_channels, base_channels, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z    = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def reconstruction_loss(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(x_hat, x, reduction="mean")


class ConvVAE(nn.Module):
    """VAE convolutionnel. Perte = MSE reconstruction + beta * KL."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        latent_dim: int = 128,
        beta: float = 1.0,
    ):
        super().__init__()
        self.beta = beta
        self.encoder = VAEEncoder(in_channels, base_channels, latent_dim)
        self.decoder = ConvDecoder(in_channels, base_channels, latent_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

    def loss(self, x: torch.Tensor, x_hat: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        l_rec = F.mse_loss(x_hat, x, reduction="mean")
        kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total = l_rec + self.beta * kl
        return total, {"loss_rec": l_rec.item(), "loss_kl": kl.item()}


def build_autoencoder(cfg: dict):
    """Retourne ConvAutoencoder (AE) ou ConvVAE selon cfg['model']."""
    model_type   = cfg.get("model", "ae_conv")
    in_ch        = 3
    base_ch      = cfg.get("base_channels", 32)
    latent_dim   = cfg.get("latent_dim", 128)
    beta         = cfg.get("beta", 1.0)

    if model_type == "vae_conv":
        return ConvVAE(in_ch, base_ch, latent_dim, beta)
    return ConvAutoencoder(in_ch, base_ch, latent_dim)
