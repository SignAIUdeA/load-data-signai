from pathlib import Path

import pandas as pd

from signai_landmarks.index import load_index, resolve_path


def test_resolve_path_builds_class_subfolder_path():
    path = resolve_path("data/landmarks", "S1V1C1M1A Clip1.npy", 1)
    assert path == Path("data/landmarks/1/S1V1C1M1A Clip1.npy")


def test_load_index_adds_resolved_path_column(tmp_path):
    landmarks_dir = tmp_path / "landmarks"
    landmarks_dir.mkdir()
    pd.DataFrame(
        {"File": ["a.npy", "b.npy"], "Class": [1, 2]}
    ).to_csv(landmarks_dir / "data_curated.csv", index=False)

    df = load_index(landmarks_dir)

    assert list(df["Path"]) == [
        landmarks_dir / "1" / "a.npy",
        landmarks_dir / "2" / "b.npy",
    ]
