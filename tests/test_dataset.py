from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from signai_landmarks.constants import FEATURE_DIM
from signai_landmarks.dataset import load_dataset

FIXTURE_ROWS = [
    # filename, class, num_frames
    ("clip_a.npy", 1, 5),
    ("clip_b.npy", 1, 8),
    ("clip_c.npy", 2, 3),
]


@pytest.fixture
def fake_landmarks_dir(tmp_path):
    landmarks_dir = tmp_path / "landmarks"
    landmarks_dir.mkdir()

    for filename, class_label, num_frames in FIXTURE_ROWS:
        class_dir = landmarks_dir / str(class_label)
        class_dir.mkdir(exist_ok=True)
        rng = np.random.default_rng(seed=hash(filename) % (2**32))
        array = rng.random((num_frames, FEATURE_DIM))
        np.save(class_dir / filename, array)

    pd.DataFrame(
        {
            "File": [row[0] for row in FIXTURE_ROWS],
            "Class": [row[1] for row in FIXTURE_ROWS],
        }
    ).to_csv(landmarks_dir / "data_curated.csv", index=False)

    return landmarks_dir


def test_load_dataset_shapes_and_dtypes(fake_landmarks_dir):
    X, y = load_dataset(fake_landmarks_dir)

    assert X.shape == (len(FIXTURE_ROWS),)
    assert X.dtype == object
    assert y.shape == (len(FIXTURE_ROWS),)
    assert np.issubdtype(y.dtype, np.integer)


def test_load_dataset_preserves_ragged_lengths(fake_landmarks_dir):
    X, _ = load_dataset(fake_landmarks_dir)

    for sequence, (_, _, num_frames) in zip(X, FIXTURE_ROWS):
        assert sequence.shape == (num_frames, FEATURE_DIM)


def test_load_dataset_label_alignment(fake_landmarks_dir):
    X, y = load_dataset(fake_landmarks_dir)

    for i, (filename, class_label, _) in enumerate(FIXTURE_ROWS):
        assert y[i] == class_label
        expected = np.load(fake_landmarks_dir / str(class_label) / filename)
        assert np.array_equal(X[i], expected)


def test_load_dataset_row_count_matches_csv(fake_landmarks_dir):
    X, y = load_dataset(fake_landmarks_dir)

    assert len(X) == len(y) == len(FIXTURE_ROWS)


REAL_LANDMARKS_DIR = Path(__file__).resolve().parent.parent / "data" / "landmarks"


@pytest.mark.skipif(
    not (REAL_LANDMARKS_DIR / "data_curated.csv").exists(),
    reason="real dataset not present locally (data/ is gitignored)",
)
def test_load_dataset_smoke_real_data():
    X, y = load_dataset(REAL_LANDMARKS_DIR)

    assert X.shape == (1500,)
    assert y.shape == (1500,)
    assert set(np.unique(y)) == {1, 2, 3, 4, 5}
    assert X[0].shape[1] == FEATURE_DIM
