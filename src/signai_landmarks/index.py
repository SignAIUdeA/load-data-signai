from pathlib import Path

import pandas as pd

from signai_landmarks.constants import CLASS_COLUMN, CSV_FILENAME, FILE_COLUMN


def resolve_path(landmarks_dir: str | Path, filename: str, class_label: int) -> Path:
    return Path(landmarks_dir) / str(class_label) / filename


def load_index(landmarks_dir: str | Path) -> pd.DataFrame:
    landmarks_dir = Path(landmarks_dir)
    df = pd.read_csv(landmarks_dir / CSV_FILENAME)
    df["Path"] = [
        resolve_path(landmarks_dir, filename, class_label)
        for filename, class_label in zip(df[FILE_COLUMN], df[CLASS_COLUMN])
    ]
    return df
