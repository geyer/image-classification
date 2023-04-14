"""Runs a single episode to compare step durations across batch sizes.
"""

import ml_collections
import numpy as np
from configs.trial_config import trial_config


def get_config():
    config = ml_collections.ConfigDict()
    config.experiment_name = 'mlp_batch_sizes'
    config.trials = []
    for param in [64, 128, 256, 512, 1024]:
        trial = trial_config('MLP')
        trial.epochs = 1
        trial.batch_size = param
        trial.trial_name = f'batch_size_{param}'
        config.trials.append(trial)
    config.lock()
    return config
