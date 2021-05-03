import os
import shutil
from pathlib import Path
from typing import Dict

import yaml


def get_config(filename: Path) -> Dict:
    with open(filename, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def remove_dir(path: Path):
    """ param <path> could either be relative or absolute. """
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    else:
        raise ValueError("file {} is not a file or dir.".format(path))
