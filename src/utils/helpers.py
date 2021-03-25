from pathlib import Path
from typing import Dict

import yaml


def get_config(filename: Path) -> Dict:
    with open(filename, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config
