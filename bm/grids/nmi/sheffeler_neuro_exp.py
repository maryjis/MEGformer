#.

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
        'audio_mous'
    ]

    # Results from Table 2.
    with launcher.job_array():
        for seed, dset in product(seeds, audio_sets):
            sub = launcher.bind({'dset.selections': [dset],'dset.bandpass': True, 'dset.bandpass_high': 0.1, 'dset.bandpass_lower' : 40.0}, seed=seed)
            sub()
            # sub({'optim.name': 'adamw', 'optim.weight_decay': 0.1})
            sub({'dset.tmin' : -0.5 , 'dset.tmax': 2.5})
            sub({'dset.tmin' : -0.75 , 'dset.tmax': 3.25 })
            sub({'dset.tmin' : -1.25 , 'dset.tmax': 5.75})
            sub({'dset.tmin' : -0.5 , 'dset.tmax': 9.5})
            sub({'dset.tmin' : -1.75 , 'dset.tmax': 8.25})
            # sub({'dset.tmin' : -1.75 , 'dset.tmax': 1.75})
            # sub({'optim.shuffle' : True})

# -0.25 - 0.75 - 1c
# -0.5 - 2.5 - 3c
# -0.75 - 3.25 - 4c
# -1.25 - 5.75 - 7c
# -1.5 - 6.50 - 8c
# -1.75 - 8.25 - 10c