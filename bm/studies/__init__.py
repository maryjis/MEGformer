#.

# flake8: noqa
from .api import Recording
from .api import register
from .api import from_selection

# all studies should be imported so as to populate the recordings dictionary
from . import schoffelen2019  # noqa
from . import gwilliams2022  # noqa
from . import broderick2019  # noqa
from . import fake  # noqa
from . import brennan2019  # noqa
