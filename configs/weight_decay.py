"""Find best weight decay configuration.

Aims to find a regularization to prevent overfitting. Uses learning rates found
for fast overfitting.
"""

import ml_collections
import numpy as np
from configs.trial_config import trial_config


def get_config():
    config = ml_collections.ConfigDict()
    config.experiment_name = 'weight_decay'
    config.trials = []
    for weight_decay in np.logspace(-5, -1, num=5):
        for lr, momentum in [(0.1, 0.9), (0.01, 0.98)]:
            trial = trial_config()
            trial.epochs = 200
            trial.weight_decay = weight_decay
            trial.learning_rate = lr
            trial.momentum = momentum
            trial.trial_name = f'lr{lr}_m{momentum}_weight_decay{weight_decay}'
            config.trials.append(trial)
    config.lock()
    return config
