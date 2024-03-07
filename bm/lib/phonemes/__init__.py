#.

import json
from pathlib import Path

ph_dict: dict = {}
dir_path = Path(__file__).parent

with open(dir_path / "phonemes.json", 'r') as f:
    ph_dict = json.load(f)
