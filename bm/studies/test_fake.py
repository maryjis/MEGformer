#.

import mne
import pandas as pd
from . import fake


def test_fake_recording() -> None:
    recording = fake.FakeRecording('sub-A2002')
    assert isinstance(recording.events(), pd.DataFrame)
    assert isinstance(recording.raw(), mne.io.RawArray)
