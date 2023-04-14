"""Use dropout as regularization.

Find dropout probabilities to avoid overfitting. Optimizer parameters were
chosen for fast overfitting without regularization.
"""

import ml_collections
import numpy as np
from configs.trial_config import trial_config


def get_config():
    config = ml_collections.ConfigDict()
    config.experiment_name = 'mlp_dropout'
    config.trials = []
    for p_dropout in [None, 0.1, 0.2, 0.3, 0.4, 0.5]:
        trial = trial_config('MLP')
        trial.epochs = 500
        trial.learning_rate = 0.1
        trial.momentum = 0.9
        trial.model_args.p_dropout = p_dropout
        trial.trial_name = f'dropout_{p_dropout}'
        config.trials.append(trial)
    config.lock()
    return config
