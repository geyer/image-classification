"""Use dropout as regularization.

Find dropout probabilities to avoid overfitting. Optimizer parameters were
chosen for fast overfitting without regularization.
"""

import ml_collections
import numpy as np
from configs.trial_config import trial_config


def get_config():
    config = ml_collections.ConfigDict()
    config.experiment_name = 'dropout'
    config.trials = []
    for p_dropout in [0.1, 0.2, 0.5, None]:
        for lr, momentum in [(0.1, 0.9), (0.01, 0.98)]:
            trial = trial_config()
            trial.epochs = 200
            trial.learning_rate = lr
            trial.momentum = momentum
            trial.model_args.p_dropout = p_dropout
            trial.trial_name = f'lr{lr}_m{momentum}_dropout{p_dropout}'
            config.trials.append(trial)
    config.lock()
    return config
