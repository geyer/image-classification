
from torch import nn
from torch.nn import functional as F
import torch
from einops.layers.torch import Rearrange


class Mlp(nn.Module):
    """Simple MLP with optional dropout after each layer."""

    def __init__(self, dims, p_dropout=None):
        """An MLP for the given sequence of hidden layer dimensions."""
        super().__init__()
        dims = (3 * 32 * 32,) + dims
        self.flatten = nn.Flatten()
        layers = []
        for n_in, n_out in zip(dims, dims[1:]):
            layers.extend([
                nn.Linear(n_in, n_out),
                nn.ReLU(),
            ])
            if p_dropout is not None:
                layers.append(
                    nn.Dropout(p=p_dropout))
        layers.append(nn.Linear(dims[-1], 10))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.layers(x)
        return logits


class ResidualBlock(nn.Module):
    def __init__(self, n_channels, double_dim=False):
        super().__init__()
        # Use projection for input layer when changing dimensions.
        first_stride = 1
        in_channels = n_channels
        self.projection = None
        if double_dim:
            first_stride = 2
            in_channels = n_channels // 2
            self.projection = nn.Conv2d(n_channels // 2, n_channels,
                                        kernel_size=1, stride=2)

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, n_channels,
                      kernel_size=3, stride=first_stride, padding=1),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(),
            nn.Conv2d(n_channels, n_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_channels),
        )

    def forward(self, x):
        y = self.layers(x)
        if self.projection is not None:
            x = self.projection(x)
        return F.relu(y + x)


class ResidualNet(nn.Module):
    """Mini residual net."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # Size 32x32
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            ResidualBlock(16),
            ResidualBlock(16),
            # Scale to 16x16
            ResidualBlock(32, double_dim=True),
            ResidualBlock(32),
            # Scale to 8x8
            ResidualBlock(64, double_dim=True),
            ResidualBlock(64),
            nn.AvgPool2d(kernel_size=8),
            nn.Flatten(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.layers(x)


class MlpBlock(nn.Module):
    def __init__(self, outer_dim, inner_dim, p_dropout=None):
        super().__init__()
        self.mlp1 = nn.Linear(outer_dim, inner_dim)
        self.dropout1 = nn.Dropout(p_dropout) if p_dropout else None
        self.mlp2 = nn.Linear(inner_dim, outer_dim)
        self.dropout2 = nn.Dropout(p_dropout) if p_dropout else None

    def forward(self, x):
        x = self.mlp1(x)
        if self.dropout1:
            x = self.dropout1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        if self.dropout2:
            x = self.dropout2(x)
        return x


class MixerBlock(nn.Module):
    def __init__(self, n_tokens, n_channels, tokens_mlp_dim, channels_mlp_dim,
                 p_dropout):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(n_channels)
        self.token_mixer = MlpBlock(n_tokens, tokens_mlp_dim, p_dropout)
        self.layer_norm2 = nn.LayerNorm(n_channels)
        self.channel_mixer = MlpBlock(n_channels, channels_mlp_dim, p_dropout)

    def forward(self, x):
        y = self.layer_norm1(x)
        y = torch.transpose(y, -1, -2)
        y = self.token_mixer(y)
        y = torch.transpose(y, -1, -2)
        x = x + y
        y = self.layer_norm2(x)
        y = self.channel_mixer(y)
        return x + y


class MlpMixer(nn.Module):
    def __init__(self, n_tokens, n_channels, tokens_mlp_dim, channels_mlp_dim, patch_size, n_blocks,
                 p_dropout=None):
        super().__init__()
        # Projects image into sequence of tokens.
        self.projection = nn.Conv2d(
            3, n_channels, kernel_size=patch_size, stride=patch_size)
        # Change to batch x token x channel order.
        layers = [Rearrange('b c h w -> b (h w) c')]
        for _ in range(n_blocks):
            layers.append(MixerBlock(n_tokens, n_channels,
                          tokens_mlp_dim, channels_mlp_dim,
                          p_dropout))
        self.layers = nn.Sequential(*layers)
        self.layer_norm = nn.LayerNorm(n_channels)
        self.final = nn.Linear(n_channels, 10)
        nn.init.zeros_(self.final.weight)

    def forward(self, x):
        x = self.projection(x)
        x = self.layers(x)
        x = self.layer_norm(x)
        # Global average pooling along channel dimension.
        x = torch.mean(x, dim=-2)
        return self.final(x)
