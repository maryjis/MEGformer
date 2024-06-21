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
            sub({'dset.tmin' : -2.5 , 'dset.tmax': 7.5})
            sub({'optim.shuffle' : True})
            sub({'simpleconv.conv_dropout' : 0.2 , 'simpleconv.depth': 6})
            sub({'clip.probabilities': True})
            sub({'simpleconv.strides': [2,1,1,1,1,1,1,1,1,1], 
                 'simpleconv.kernel_size' : [7,7,5,5,5,3,3,3,3,3], 
                 'simpleconv.padding' : [0,0,0,0,0,0,0,0,0,0,0,0], 
                 'simpleconv.seq_len' : 361,
                 'simpleconv.auto_padding' : False, 
                 'dset.features_params.Wav2VecTransformer.is_interpolate' : False , 'simpleconv.dilation_period' : 1, 'task.offset_meg_ms' : 0})
            sub({'simpleconv.strides': [1,1,1,1,1,1,1,1,1,1], 
                 'simpleconv.kernel_size' : [3,3,3,3,3,3,3,3,3,3], 
                 'simpleconv.padding' : [1,1,1,1,1,1,1,1,1,1], 
                 'simpleconv.seq_len' : 361,
                 'simpleconv.auto_padding' : False, 
                 'dset.features_params.Wav2VecTransformer.is_interpolate' : False , 'simpleconv.dilation_period' : 1, 'task.offset_meg_ms' : 0, 'dset.features_params.Wav2VecTransformer.size' :361})
            sub({'simpleconv.strides': [1,1,1,1,1,1,1,1,1,1], 
                 'simpleconv.kernel_size' : [3,3,3,3,3,3,3,3,3,3], 
                 'simpleconv.padding' : [1,1,1,1,1,1,1,1,1,1], 
                 'simpleconv.seq_len' : 361,
                 'simpleconv.auto_padding' : False, 
                 'dset.features_params.Wav2VecTransformer.is_interpolate' : False , 'simpleconv.dilation_period' : 5, 'task.offset_meg_ms' : 0, 'dset.features_params.Wav2VecTransformer.size' :361})
            sub({'dset.sample_rate' : 240})
            sub({'simpleconv.dilation_period' : 1})
            sub({'simpleconv.seq_len' : 361})
            sub({'simpleconv.seq_len' : 361, 'dset.features_params.Wav2VecTransformer.is_interpolate' : False ,'dset.features_params.Wav2VecTransformer.size' :361 })
            sub({'simpleconv.is_deformable_conv' : True, 'simpleconv.conv_dropout' : 0.2 , 'simpleconv.depth': 6})
            sub({'simpleconv.is_deformable_conv' : True, 'simpleconv.conv_dropout' : 0.2 , 'simpleconv.depth': 6, 'simpleconv.dilation_period' : 3})
            sub({'dset.features_params.Wav2VecTransformer.layers' : [1]})
            sub({'dset.features_params.Wav2VecTransformer.layers' : [14]})
            sub({'dset.features_params.Wav2VecTransformer.layers' : [18]})
            sub({'dset.features_params.Wav2VecTransformer.layers' : [9]})
            sub({'dset.condition': 3.0})
            sub({'is_sound': False})
            sub({'simpleconv.conv_dropout' : 0.2 , 'simpleconv.depth': 6, 'is_sound': True})
            sub({'is_sound': False})
            sub({'dset.tmin' : -2.5 , 'dset.tmax': 7.5, 'is_sound': True})
            sub({'dset.tmin' : -1.25 , 'dset.tmax': 5.75, 'is_sound': True})
            sub({'dset.tmin' : -0.75 , 'dset.tmax': 3.25, 'is_sound': True})
            sub({'dset.tmin' : -1.0 , 'dset.tmax': 1.5, 'is_sound': True})
            
