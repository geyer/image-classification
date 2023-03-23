"""Find optimizer parameters that allow fast training.

Aims to find optimizer parameters for fast overfitting for the given model.
"""

import ml_collections
import numpy as np
from configs.trial_config import trial_config


def get_config():
    config = ml_collections.ConfigDict()
    config.experiment_name = 'optimizer_params'
    params = [{'learning_rate': float(lr), 'momentum': float(m)}
              for lr in np.logspace(-4, 0, num=5)
              for m in (0, 0.5, 0.9, 0.98)
              ]
    config.trials = []
    for param in params:
        trial = trial_config()
        trial.update(**param)
        lr, momentum = param['learning_rate'], param['momentum']
        trial.trial_name = f'lr{lr}_m{momentum}'
        config.trials.append(trial)
    config.lock()
    return config
