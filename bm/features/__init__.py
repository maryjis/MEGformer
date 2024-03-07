#.
"""Features describe how to transform the input sparse annotation, e.g.
words or wav file names, into actual dense features for training neural network,
or to be used as targets for the contrastive loss.
"""
from .base import FeaturesBuilder, Feature  # noqa
from . import basic  # noqa
from . import audio  # noqa
from . import embeddings  # noqa
