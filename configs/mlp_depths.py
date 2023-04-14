"""Compare different model depths (number of layers).

Uses a fixed hidden dimension across all layers.  Does not tune hyperparameters per depth.
"""

import ml_collections
import numpy as np
from configs.trial_config import trial_config


def get_config():
    config = ml_collections.ConfigDict()
    config.experiment_name = 'mlp_depths_deep'
    config.trials = []
    for depth in [8, 4, 2]:
        trial = trial_config('MLP')
        trial.epochs = 200
        trial.learning_rate = 0.1
        trial.momentum = 0.9
        trial.weight_decay = 1e-4
        trial.model_args.p_dropout = 0.1
        trial.model_args.dims = tuple([512] * depth)
        trial.trial_name = f'depth_{depth}'
        config.trials.append(trial)
    config.lock()
    return config
