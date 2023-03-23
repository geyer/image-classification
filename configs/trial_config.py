"""Default trial config.

To be modified for experiments.  Sets all trial fields with default parameters.
"""

import ml_collections


def trial_config():
    trial = ml_collections.ConfigDict()
    trial.batch_size = 1024
    trial.epochs = 100
    trial.checkpoint_after_episodes = 50
    trial.learning_rate = 0.1
    trial.momentum = 0.9
    trial.weight_decay = 0.0
    trial.trial_name = 'default'
    trial.model_type = 'MLP'
    trial.dropout_rate = None
    # All trial fields must be specified here.
    trial.lock()
    return trial
