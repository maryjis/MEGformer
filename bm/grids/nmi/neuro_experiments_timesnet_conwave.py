# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product  # noqa
from .._explorers import ClipExplorer
from ...train import main  # noqa
"""Results from the main table."""


@ClipExplorer
def explorer(launcher):
    launcher.slurm_(
        gpus=2, mem_per_gpu=200,
        partition="learnlab",
        constraint="volta32gb",
    )
    launcher.bind_({
        'model': 'timesnet',
        'feature_model' : 'conv_wave'
    })

    seeds = [2036, 2037, 2038]
    audio_sets = [
        'gwilliams2022'
    ]

    # Results from Table 2.
    with launcher.job_array():
        for seed, dset in product(seeds, audio_sets):
            sub = launcher.bind({'dset.selections': [dset],'dset.bandpass': True, 'dset.bandpass_high': 0.1, 'dset.bandpass_lower' : 40.0}, seed=seed)
            sub()
            sub({'feature_model_params.dropout_value' : 0.3,
                 'feature_model_params.n_out_channels' : 64, 'timesnet.flatten_out_channels': 64})
            sub({'feature_model_params.dropout_value' : 0.3,
                 'feature_model_params.n_out_channels' : 64, 'timesnet.flatten_out_channels': 64, 'timesnet.d_model': 64,'timesnet.d_ff': 128})
            sub({'feature_model_params.dropout_value' : 0.3,
                 'feature_model_params.n_out_channels' : 64, 'timesnet.flatten_out_channels': 64, 'timesnet.d_model': 64,'timesnet.d_ff': 128, 'timesnet.subject_layers': True})
            sub({'feature_model_params.dropout_value' : 0.3,
                 'feature_model_params.n_out_channels' : 64, 'timesnet.flatten_out_channels': 64, 'timesnet.d_model': 64,
                 'timesnet.d_ff': 128, 'timesnet.subject_layers': True, 'timesnet.merger': True})
            sub({'feature_model_params.dropout_value' : 0.3,
                 'feature_model_params.n_out_channels' : 64, 'timesnet.flatten_out_channels': 64, 'timesnet.d_model': 128,
                 'timesnet.d_ff': 256, 'timesnet.subject_layers': True, 'timesnet.merger': True})
