# Image classification on cifar10

Some scripts to experiment with image classification models on the cifar10
dataset. The script `run_training.py` runs a single trial as specified by the
configuration `configs/trial_config.py`.  The script `run_experiment.py` runs a
sequence of trials as specified by one of the experiment configs in `configs/`.

When running an individual training run, the output (checkpoints and
tensorboard) is written to `./trials/<trial_name>`.  For experiments, the output
is at `./experiments/<experiment_name>/`.  This allows for an easy comparison of
the trials runs within an experiment.

The current models are a two layer MLP, and a small residual net. The existing
experiment configurations are to find

* optimizer parameters (SGD) for overfitting quickly,
* regularization parameters (dropout for MLP, weight decay for both models) to
  find a non-overfitting solution.
