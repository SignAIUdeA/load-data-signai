from pathlib import Path

import numpy as np

from signai_landmarks.index import load_index
from signai_landmarks.landmarks import load_landmark_file


def load_dataset(landmarks_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    pairs = load_index(landmarks_dir)
    sequences = [load_landmark_file(path) for path, _ in pairs]
    X = np.array(sequences, dtype=object)
    y = np.array([class_label for _, class_label in pairs], dtype=int)
    return X, y
