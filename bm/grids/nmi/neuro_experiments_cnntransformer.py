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
        'model': 'clip_cnntransformer',
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
            sub({'is_sound': True})
            sub({'is_sound': True})
            sub({ 'cnntransformer.depth' : 4,
                 'cnntransformer.conv_dropout' : 0.2})
            sub({ 'cnntransformer.depth' : 4,
                 'cnntransformer.conv_dropout' : 0.2})
            sub({ 'cnntransformer.depth' : 4,
                 'cnntransformer.depth_transformer' : 3})
            
            