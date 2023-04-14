"""Find optimizer parameters for fast convergence (overfitting).

Use SGD with/without momentum.
"""

import ml_collections
import numpy as np
from configs.trial_config import trial_config


def get_config():
    config = ml_collections.ConfigDict()
    config.experiment_name = 'mlp_optimizer'
    params = [(lr, m)
              for lr in np.logspace(-4, 0, num=5)
              for m in (0, 0.8, 0.9, 0.95)]
    config.trials = []
    for lr, m in params:
        trial = trial_config('MLP')
        trial.epochs = 20
        trial.learning_rate = lr
        trial.momentum = m
        trial.trial_name = f'lr_{lr}_m_{m}'
        config.trials.append(trial)
    config.lock()
    return config
