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
        'model': 'clip_transformer',
        'feature_model' : 'conv_wave'
    })

    seeds = [2036, 2037, 2038]
    audio_sets = [
        'audio_mous'
    ]

    # Results from Table 2.
    with launcher.job_array():
        for seed, dset in product(seeds, audio_sets):
            sub = launcher.bind({'dset.selections': [dset],'dset.bandpass': True, 'dset.bandpass_high': 0.1, 'dset.bandpass_lower' : 40.0}, seed=seed)
            sub()
            sub({'feature_model_params.dropout_value' : 0.3,'feature_model_params.n_out_channels' : 8,
                'simpletransformer.subject_layers' : True, 'simpletransformer.positional_embedding': True, 'simpletransformer.depth' : 8, 'simpletransformer.nhead' : 8, 'simpletransformer.model_type':'bigbird'})
            sub({'feature_model_params.dropout_value' : 0.3,'feature_model_params.n_out_channels' : 8,
                'simpletransformer.subject_layers' : True, 'simpletransformer.positional_embedding': True, 'simpletransformer.depth' : 8, 'simpletransformer.nhead' : 8, 'simpletransformer.model_type':'bigbird', 'simpletransformer.attention_type': 'original_full'})
            sub({'feature_model_params.dropout_value' : 0.3,'feature_model_params.n_out_channels' : 8,
                'simpletransformer.subject_layers' : True, 'simpletransformer.positional_embedding': True, 'simpletransformer.depth' : 8, 'simpletransformer.nhead' : 8, 'simpletransformer.model_type':'bigbird', 'simpletransformer.block_size': 8})
            sub({'feature_model_params.dropout_value' : 0.3,'feature_model_params.n_out_channels' : 8,
                'simpletransformer.subject_layers' : True, 'simpletransformer.positional_embedding': True, 'simpletransformer.depth' : 8, 'simpletransformer.nhead' : 8, 'simpletransformer.model_type':'bigbird', 'simpletransformer.block_size': 2})
            sub({'feature_model_params.dropout_value' : 0.3,'feature_model_params.n_out_channels' : 8,
                'simpletransformer.subject_layers' : True, 'simpletransformer.positional_embedding': True, 'simpletransformer.depth' : 8, 'simpletransformer.nhead' : 8, 'simpletransformer.model_type':'bigbird', 'simpletransformer.num_random_blocks': 2})
            sub({'feature_model_params.dropout_value' : 0.3,'feature_model_params.n_out_channels' : 8,
                'simpletransformer.subject_layers' : True, 'simpletransformer.positional_embedding': True, 'simpletransformer.depth' : 8, 'simpletransformer.nhead' : 8, 'simpletransformer.model_type':'bigbird', 'simpletransformer.num_random_blocks': 5})

            sub({'feature_model_params.dropout_value' : 0.3,'feature_model_params.n_out_channels' : 8,
                'simpletransformer.subject_layers' : True, 'simpletransformer.positional_embedding': True, 'simpletransformer.depth' : 8, 'simpletransformer.nhead' : 8, 'simpletransformer.model_type':'bigbird', 'simpletransformer.block_size': 2, 'simpletransformer.merger': True
                })



            sub({'feature_model_params.dropout_value' : 0.3,'feature_model_params.n_out_channels' : 8,
                'simpletransformer.subject_layers' : True, 'simpletransformer.positional_embedding': True, 'simpletransformer.depth' : 6, 'simpletransformer.nhead' : 6, 'simpletransformer.model_type':'reformer', 'simpletransformer.block_size': 2
                })
            



         
            