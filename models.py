
from torch import nn
from torch.nn import functional as F

class Mlp(nn.Module):
    """Simple MLP with optional dropout after each layer."""

    def __init__(self, dropout_rate=None):
        super().__init__()
        self.flatten = nn.Flatten()
        if dropout_rate is not None:
            self.layers = nn.Sequential(
                nn.Linear(3*32*32, 512),
                nn.Dropout(p=dropout_rate),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.Dropout(p=dropout_rate),
                nn.ReLU(),
                nn.Linear(512, 10),
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(3*32*32, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10),
            )

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
