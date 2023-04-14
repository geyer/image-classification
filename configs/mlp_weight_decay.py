"""Adds weight decay configuration while using dropout.

Aims to find a regularization to prevent overfitting. Uses learning rates found
for fast overfitting.
"""

import ml_collections
import numpy as np
from configs.trial_config import trial_config


def get_config():
    config = ml_collections.ConfigDict()
    config.experiment_name = 'mlp_weight_decay'
    config.trials = []
    for param in np.logspace(-5, -1, num=5):
        trial = trial_config('MLP')
        trial.epochs = 200
        trial.learning_rate = 0.1
        trial.momentum = 0.9
        trial.weight_decay = param
        trial.model_args.p_dropout = 0.1
        trial.trial_name = f'weight_decay_{param}'
        config.trials.append(trial)
    config.lock()
    return config
