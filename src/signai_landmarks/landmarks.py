from pathlib import Path

import numpy as np


def load_landmark_file(path: str | Path) -> np.ndarray:
    return np.load(path)
