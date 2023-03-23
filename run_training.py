"""Trains a classifier on the cifar10 dataset.

Takes a training run configuration to simplify running experiments.

Writes training curves to tensorboard to simplify comparisons between experiments.
"""
from tqdm import tqdm
import ml_collections
import time
import os
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from configs.trial_config import trial_config


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


def load_dataset(root='~/datasets', batch_size=64):
    """Returns dataloaders for (train, test) sets."""
    train_data = datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    test_data = datasets.CIFAR10(
        root=root,
        train=False,
        download=False,
        transform=transforms.ToTensor(),
    )

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    return (train_loader, test_loader)


def train_with_config(config, base_dir=None):
    if base_dir is None:
        base_dir = './trials'
    os.makedirs(os.path.join(base_dir, 'checkpoints'), exist_ok=True)
    writer = SummaryWriter(os.path.join(base_dir, 'runs', config.trial_name))

    train_data, test_data = load_dataset(
        '~/datasets', batch_size=config.batch_size)

    # Select and configure the model.
    device = torch.device('cuda')
    if config.model_type == 'MLP':
        model = Mlp(dropout_rate=config.dropout_rate)
    elif config.model_type == 'ResNet':
        model = ResidualNet()
    model = model.to(
        device, memory_format=torch.channels_last, non_blocking=True)

    with torch.no_grad():
        sample_batch, _ = next(iter(train_data))
        writer.add_graph(model, sample_batch.to(
            device, memory_format=torch.channels_last))

    @torch.no_grad()
    def evaluate_model(n_steps):
        model.eval()
        losses = []
        n_right, n_total = 0, 0
        for batch, labels in test_data:
            batch = batch.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(batch)
            preds = torch.argmax(torch.sigmoid(logits), dim=1)
            n_right += torch.sum(preds == labels).item()
            n_total += batch.shape[0]
            loss = F.cross_entropy(logits, labels)
            losses.append(loss.item())
        writer.add_scalar('Loss/eval', sum(losses) / len(losses), n_steps)
        writer.add_scalar('Metrics/accuracy', n_right / n_total, n_steps)

    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate,
                          momentum=config.momentum, weight_decay=config.weight_decay)
    n_steps = 0
    # Run an evaluation on the initalized model.
    evaluate_model(n_steps)
    for epoch in tqdm(range(config.epochs)):
        model.train()
        for batch, labels in train_data:
            start = time.perf_counter_ns()
            batch = batch.to(
                device, memory_format=torch.channels_last, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            stop = time.perf_counter_ns()
            duration_ms = (stop - start) / 1e6
            writer.add_scalar('Duration/step_ms', duration_ms, n_steps)
            writer.add_scalar('Duration/example_ms',
                              duration_ms / batch.shape[0], n_steps)
            writer.add_scalar('Loss/train', loss.item(), n_steps)
            with torch.no_grad():
                grad_norm = 0
                for param in model.parameters():
                    grad_norm += torch.linalg.norm(param.grad.data).cpu() ** 2
                grad_norm = torch.sqrt(grad_norm)
            writer.add_scalar('grad_norm', grad_norm.item(), n_steps)
            n_steps += 1

        if epoch % config.checkpoint_after_episodes == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            },
                os.path.join(base_dir, 'checkpoints',
                             f'{config.trial_name}_epoch_{epoch}'))

        # Evaluate on test data after each episode.
        evaluate_model(n_steps)


if __name__ == '__main__':
    config = trial_config()
    train_with_config(config)
