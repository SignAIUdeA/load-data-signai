import pytest

from signai_landmarks.index import class_label_from_dir, load_index


def test_class_label_from_dir_parses_prefix():
    assert class_label_from_dir("sign_03") == 3


def test_class_label_from_dir_rejects_unprefixed_name():
    with pytest.raises(ValueError):
        class_label_from_dir("03")


def test_load_index_walks_sign_folders(tmp_path):
    landmarks_dir = tmp_path / "landmarks"
    sign_01 = landmarks_dir / "sign_01"
    sign_02 = landmarks_dir / "sign_02"
    sign_01.mkdir(parents=True)
    sign_02.mkdir(parents=True)

    (sign_01 / "S01_M02_V01_C01_Clip002.npy").touch()
    (sign_01 / "S01_M01_V01_C01_Clip001.npy").touch()
    (sign_02 / "S02_M01_V01_C02_Clip301.npy").touch()

    pairs = load_index(landmarks_dir)

    assert pairs == [
        (sign_01 / "S01_M01_V01_C01_Clip001.npy", 1),
        (sign_01 / "S01_M02_V01_C01_Clip002.npy", 1),
        (sign_02 / "S02_M01_V01_C02_Clip301.npy", 2),
    ]


def test_load_index_ignores_non_sign_dirs_and_non_npy_files(tmp_path):
    landmarks_dir = tmp_path / "landmarks"
    sign_01 = landmarks_dir / "sign_01"
    sign_01.mkdir(parents=True)
    (sign_01 / "S01_M01_V01_C01_Clip001.npy").touch()
    (sign_01 / "notes.txt").touch()
    (landmarks_dir / "other_stuff").mkdir()
    (landmarks_dir / "README.md").touch()

    pairs = load_index(landmarks_dir)

    assert pairs == [(sign_01 / "S01_M01_V01_C01_Clip001.npy", 1)]


def test_load_index_raises_on_missing_or_empty_landmarks_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_index(tmp_path / "does_not_exist")

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        load_index(empty_dir)
