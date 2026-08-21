from pathlib import Path

import numpy as np

from signai_landmarks.constants import CLASS_COLUMN
from signai_landmarks.index import load_index
from signai_landmarks.landmarks import load_landmark_file


def load_dataset(landmarks_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    df = load_index(landmarks_dir)
    sequences = [load_landmark_file(path) for path in df["Path"]]
    X = np.array(sequences, dtype=object)
    y = df[CLASS_COLUMN].to_numpy(dtype=int)
    return X, y
