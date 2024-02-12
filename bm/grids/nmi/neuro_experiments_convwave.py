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
        'model': 'clip_conv',
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
            sub({'simpleconv.avg_pool_out': True, 'simpleconv.strides' : [2,1,2,1,2,1,2,1,2,1], 
                 'simpleconv.complex_out' : False, 'simpleconv.adaptive_pooling_size' : 1, 'simpleconv.dilation_period' : 1})
            sub({'simpleconv.flatten_out': True, 'simpleconv.complex_out' : False})
            sub({'simpleconv.flatten_out': True, 'simpleconv.complex_out' : False, 'feature_model_params.dropout_value' : 0.3})
            sub({'simpleconv.flatten_out': True, 'simpleconv.complex_out' : False, 'feature_model_params.dropout_value' : 0.3,
                 'feature_model_params.n_out_channels' : 256, 'simpleconv.flatten_out_channels': 256})
            sub({'simpleconv.flatten_out': True, 'simpleconv.complex_out' : False, 'feature_model_params.dropout_value' : 0.3,
                 'feature_model_params.n_out_channels' : 128, 'simpleconv.flatten_out_channels': 128})
            sub({'simpleconv.flatten_out': True, 'simpleconv.complex_out' : False, 'feature_model_params.dropout_value' : 0.3,
                 'feature_model_params.n_out_channels' : 64, 'simpleconv.flatten_out_channels': 64})
            sub({'simpleconv.flatten_out': True, 'simpleconv.complex_out' : False, 'feature_model_params.dropout_value' : 0.3,
                 'feature_model_params.n_out_channels' : 64, 'simpleconv.flatten_out_channels': 64, 'simpleconv.conv_dropout' : 0.2})
            sub({'simpleconv.flatten_out': True, 'simpleconv.complex_out' : False, 'feature_model_params.dropout_value' : 0.3,
                 'feature_model_params.n_out_channels' : 32, 'simpleconv.flatten_out_channels': 32})
            sub({'simpleconv.flatten_out': True, 'simpleconv.complex_out' : False, 'feature_model_params.dropout_value' : 0.3,
                 'feature_model_params.n_out_channels' : 1, 'simpleconv.flatten_out_channels': 1})
            sub({'simpleconv.flatten_out': True, 'simpleconv.complex_out' : False, 'feature_model_params.dropout_value' : 0.3,
                 'feature_model_params.n_out_channels' : 64, 'simpleconv.flatten_out_channels': 64, 'simpleconv.conv_dropout' : 0.2 , 'simpleconv.depth': 8})
            sub({'simpleconv.flatten_out': True, 'simpleconv.complex_out' : False, 'feature_model_params.dropout_value' : 0.3,
                 'feature_model_params.n_out_channels' : 64, 'simpleconv.flatten_out_channels': 64, 'simpleconv.conv_dropout' : 0.2 , 'simpleconv.depth': 6})
            sub({'simpleconv.flatten_out': True, 'simpleconv.complex_out' : False, 'feature_model_params.dropout_value' : 0.3,
                 'feature_model_params.n_out_channels' : 64, 'simpleconv.flatten_out_channels': 64, 'simpleconv.conv_dropout' : 0.2 , 'simpleconv.depth': 4})
            sub({'simpleconv.flatten_out': True, 'simpleconv.complex_out' : False, 'feature_model_params.dropout_value' : 0.3,
                 'feature_model_params.n_out_channels' : 32, 'simpleconv.flatten_out_channels': 32, 'simpleconv.conv_dropout' : 0.2 , 'simpleconv.depth': 6})
            sub({'simpleconv.flatten_out': True, 'simpleconv.complex_out' : False, 'feature_model_params.dropout_value' : 0.3,
                 'feature_model_params.n_out_channels' : 32, 'simpleconv.flatten_out_channels': 32, 'simpleconv.conv_dropout' : 0.2 , 
                 'simpleconv.depth': 6 , 'feature_model_params.layers_block': True, 'feature_model_params.layers_number': 1})
            sub({'simpleconv.flatten_out': True, 'simpleconv.complex_out' : False, 'feature_model_params.dropout_value' : 0.3,
                 'feature_model_params.n_out_channels' : 32, 'simpleconv.flatten_out_channels': 32, 'simpleconv.conv_dropout' : 0.2 , 
                 'simpleconv.depth': 6 , 'feature_model_params.layers_block': True, 'feature_model_params.layers_number': 1, 'feature_model_params.activation': 'gelu'})
            sub({'simpleconv.flatten_out': True, 'simpleconv.complex_out' : False, 'feature_model_params.dropout_value' : 0.3,
                 'feature_model_params.n_out_channels' : 32, 'simpleconv.flatten_out_channels': 32, 'simpleconv.conv_dropout' : 0.2 , 
                 'simpleconv.depth': 6 , 'feature_model_params.layers_block': True, 'feature_model_params.layers_number': 1,
                 'feature_model_params.activation': 'gelu', 'feature_model_params.layers_dropout': 0.6})
           
            
            
