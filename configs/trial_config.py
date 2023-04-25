"""Default trial config.

To be modified for experiments.  Sets all trial fields with default parameters.
"""

import ml_collections


def trial_config(model_type='MLP'):
    trial = ml_collections.ConfigDict()
    trial.batch_size = 1024
    trial.epochs = 100
    trial.checkpoint_after_episodes = 50
    trial.learning_rate = 0.1
    trial.momentum = 0.9
    trial.weight_decay = 0.0
    trial.gradient_clipping = None
    trial.mixup_alpha = None
    trial.lr_schedule = None
    trial.trial_name = 'default'
    trial.model_type = ''
    trial.model_args = ml_collections.ConfigDict()
    if model_type == 'MLP':
        trial.model_type = 'MLP'
        trial.model_args.dims = (512, 512)
        trial.model_args.p_dropout = None
    elif model_type == 'MlpMixer':
        trial.model_type = 'MlpMixer'
        trial.model_args.patch_size = 8
        trial.model_args.n_channels = 64
        trial.model_args.n_blocks = 8
        trial.model_args.channels_mlp_dim = 128
        trial.model_args.tokens_mlp_dim = 32
        trial.model_args.p_dropout = None
    elif model_type == 'ResNet':
        trial.model_type = 'ResNet'
    # All trial fields must be specified here.
    trial.lock()
    return trial
