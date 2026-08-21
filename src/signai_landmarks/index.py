from pathlib import Path

from signai_landmarks.constants import CLASS_FOLDER_PREFIX


def class_label_from_dir(dir_name: str) -> int:
    if not dir_name.startswith(CLASS_FOLDER_PREFIX):
        raise ValueError(f"{dir_name!r} does not start with {CLASS_FOLDER_PREFIX!r}")
    return int(dir_name[len(CLASS_FOLDER_PREFIX):])


def load_index(landmarks_dir: str | Path) -> list[tuple[Path, int]]:
    landmarks_dir = Path(landmarks_dir)
    pairs = []
    for class_dir in sorted(landmarks_dir.glob(f"{CLASS_FOLDER_PREFIX}*")):
        if not class_dir.is_dir():
            continue
        class_label = class_label_from_dir(class_dir.name)
        for npy_path in sorted(class_dir.glob("*.npy")):
            pairs.append((npy_path, class_label))

    if not pairs:
        raise FileNotFoundError(
            f"no {CLASS_FOLDER_PREFIX}*/*.npy files found under {landmarks_dir}"
        )
    return pairs
