"""Runs trials for an experiment sequentially.

Experiments are specified as an ml_collection config file.  The file should
yield a list of trial configurations.  These configurations are written (as
yaml) and a training run is started for each trial (in sequence).
"""

from absl import app
from ml_collections import config_flags
import os

from run_training import train_with_config

_CONFIG_FILE = config_flags.DEFINE_config_file('experiment_config', None,
                                               'Path to a config file.')


def main(_):
    experiment = _CONFIG_FILE.value
    print(f'Running experiment {experiment.experiment_name}'
          f' with {len(experiment.trials)} trials.')
    experiment_dir = f'./experiments/{experiment.experiment_name}'
    print(f'Writing trial configs to {experiment_dir}')
    os.makedirs(experiment_dir, exist_ok=True)
    for config in experiment.trials:
        with open(os.path.join(experiment_dir, f'{config.trial_name}.yaml'), 'w') as f:
            f.write(config.to_yaml())

    # Run all trials in sequence.
    for i, config in enumerate(experiment.trials):
        print(
            f'Running trial {config.trial_name} [{i + 1}/{len(experiment.trials)}].')
        train_with_config(config, experiment_dir)


if __name__ == '__main__':
    app.run(main)
