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
            sub({'optim.shuffle' : True})
            sub({'simpleconv.conv_dropout' : 0.2 , 'simpleconv.depth': 6})
            sub({'clip.probabilities': True})
