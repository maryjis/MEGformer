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
            sub({'optim.name': 'adamw', 'optim.weight_decay': 0.1})
            sub({'dset.tmin' : -0.2 , 'dset.tmax': 0.8})
            sub({'dset.tmin' : -0.75 , 'dset.tmax': 3.25})
            sub({'dset.tmin' : -1.25 , 'dset.tmax': 5.75})
            sub({'dset.tmin' : -1.75 , 'dset.tmax': 1.75})
            sub({'optim.shuffle' : True})
            sub({'simpletransformer.subject_layers' : True, 'simpletransformer.positional_embedding': True})
            sub({'simpletransformer.subject_layers' : True, 'simpletransformer.positional_embedding': True, 'simpletransformer.merger' : True})
            sub({'simpletransformer.subject_layers' : True, 'simpletransformer.positional_embedding': True, 'simpletransformer.merger' : True, 'simpletransformer.dim_ff' : 1024})
            sub({'simpletransformer.subject_layers' : True, 'simpletransformer.positional_embedding': True, 'simpletransformer.merger' : True , 'simpletransformer.model_type': 'logformer'})
            
            sub({'simpletransformer.subject_layers' : True, 'simpletransformer.positional_embedding': True, 'simpletransformer.merger' : True ,
                 'simpletransformer.model_type': 'logformer', 'simpletransformer.attention_window': [256,256,256,256], 'optim.lr': 3e-4, 'optim.beta2' :0.999, 'optim.eps': 1e-08})
            sub({'simpletransformer.subject_layers' : True, 'simpletransformer.positional_embedding': True, 'simpletransformer.depth' : 8, 'simpletransformer.nhead' : 8, 'dset.tmin' : -0.75 , 'dset.tmax': 3.25})
            sub({'simpletransformer.subject_layers' : True, 'simpletransformer.positional_embedding': True, 'simpletransformer.depth' : 8, 
                 'simpletransformer.nhead' : 8, 'dset.tmin' : -0.75 , 'dset.tmax': 3.25, 'optim.scheduler.name' : 'TransformerScheduler'})
            
            sub({'simpletransformer.subject_layers' : True, 'simpletransformer.positional_embedding': True, 'simpletransformer.depth' : 8, 'simpletransformer.nhead' : 8, 'is_sound' : True})
            sub({'simpletransformer.subject_layers' : True, 'simpletransformer.positional_embedding': True, 'simpletransformer.depth' : 8, 
                 'simpletransformer.nhead' : 8, 'optim.scheduler.name' : 'TransformerScheduler', 'is_sound' : True})
            sub({'simpletransformer.subject_layers' : True, 'simpletransformer.positional_embedding': True,
                 'simpletransformer.merger' : True, 'is_sound' : True})
            
