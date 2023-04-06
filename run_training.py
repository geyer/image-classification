"""Trains a classifier on the cifar10 dataset.

Takes a training run configuration to simplify running experiments.

Writes training curves to tensorboard to simplify comparisons between experiments.
"""
import os
import shutil
import time
from tqdm import tqdm

import torch
from torch import optim
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision import datasets
from torchvision import transforms

from configs.trial_config import trial_config
from models import Mlp, ResidualNet, MlpMixer


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
    checkpoint_path = os.path.join(
        base_dir, 'checkpoints', f'{config.trial_name}_latest.pt')
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    writer = SummaryWriter(os.path.join(base_dir, 'runs', config.trial_name))

    train_data, test_data = load_dataset(
        '~/datasets', batch_size=config.batch_size)

    # Initialize the training state (model, optimizer, epoch, step).
    checkpoint = None
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)

    device = torch.device('cuda')
    if config.model_type == 'MLP':
        model = Mlp(**config.model_args)
    elif config.model_type == 'ResNet':
        model = ResidualNet()
    elif config.model_type == 'MlpMixer':
        model = MlpMixer(patch_size=8, n_tokens=16, **config.model_args)

    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(
        device, memory_format=torch.channels_last, non_blocking=True)

    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate,
                          momentum=config.momentum, weight_decay=config.weight_decay)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    scheduler = None
    if config.lr_schedule == 'linear_decay':
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.01,
                             total_iters=config.epochs)
        if checkpoint is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    with torch.no_grad():
        sample_batch, _ = next(iter(train_data))
        writer.add_graph(model, sample_batch.to(
            device, memory_format=torch.channels_last))

    @torch.no_grad()
    def evaluate_model(step):
        model.eval()
        losses = []
        n_correct, n_total = 0, 0
        for batch, labels in test_data:
            batch = batch.to(
                device, memory_format=torch.channels_last, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(batch)
            preds = torch.argmax(torch.sigmoid(logits), dim=1)
            n_correct += torch.sum(preds == labels).item()
            n_total += batch.shape[0]
            loss = F.cross_entropy(logits, labels)
            losses.append(loss.item())
        writer.add_scalar('Loss/eval', sum(losses) /
                          len(losses), step)
        writer.add_scalar('Metrics/accuracy', n_correct /
                          n_total, step)

    # Number of steps and epochs passed so far.
    step, epoch = 0, 0
    if checkpoint is not None:
        step = checkpoint['step']
        epoch = checkpoint['epoch']
        print(f'Starting at epoch {epoch} at {step} steps.')

    # Run an evaluation on the initalized model.
    evaluate_model(step)
    for epoch in tqdm(range(epoch + 1, config.epochs + 1)):
        lr = config.learning_rate
        if scheduler:
            lr = scheduler.get_last_lr()[0]
        writer.add_scalar('learning_rate', lr, step)

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
            with torch.no_grad():
                grad_norm = 0
                for param in model.parameters():
                    grad_norm += torch.linalg.norm(param.grad.data).cpu() ** 2
                grad_norm = torch.sqrt(grad_norm)
            if config.gradient_clipping is not None:
                clip_grad_norm_(model.parameters(),
                                max_norm=config.gradient_clipping)
            optimizer.step()
            step += 1

            stop = time.perf_counter_ns()
            duration_ms = (stop - start) / 1e6
            writer.add_scalar('Duration/step_ms', duration_ms, step)
            writer.add_scalar('Duration/example_ms',
                              duration_ms / batch.shape[0], step)
            writer.add_scalar('Loss/train', loss.item(), step)
            writer.add_scalar('grad_norm', grad_norm.item(), step)
        if scheduler:
            scheduler.step()

        # Save training state.
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        torch.save(checkpoint,
                   checkpoint_path)
        if epoch % config.checkpoint_after_episodes == 0:
            shutil.copy(checkpoint_path,
                        os.path.join(base_dir, 'checkpoints',
                                     f'{config.trial_name}_epoch_{epoch}.pt'))

        # Evaluate on test data after each episode.
        evaluate_model(step)


if __name__ == '__main__':
    config = trial_config('MlpMixer')
    config.model_args.p_dropout = 0.1
    config.weight_decay = 0.001
    config.gradient_clipping = 1.5
    config.epochs = 300
    config.lr_schedule = 'linear_decay'
    train_with_config(config)
