"""Landmark extraction used to build the published LSC5 arrays.

Ported from ``SignAIUdeA/AI`` (commit ``b9ddcb6``),
``Modeling/Preprocessing/holistic.py`` and ``Modeling/Preprocessing/transform.py``.
The extraction logic is unchanged; only the on-frame drawing calls were removed,
because they paint on a copy of the frame that is never saved and do not affect
the landmarks.

Pipeline (one video -> one ``(T, 1662)`` float64 array):

* one MediaPipe Holistic instance per video, with ``static_image_mode=False`` and
  ``model_complexity=2``; every other parameter keeps the MediaPipe default
  (``smooth_landmarks=True``, ``refine_face_landmarks=False``,
  ``min_detection_confidence=0.5``, ``min_tracking_confidence=0.5``);
* frames are read with OpenCV until the first failed read, converted from BGR to
  RGB and passed to ``holistic.process``;
* each frame gives one row: pose (33 x [x, y, z, visibility]), face (468 x
  [x, y, z]), left hand (21 x [x, y, z]) and right hand (21 x [x, y, z]);
* a component that is not detected is filled with zeros; no imputation,
  normalization, resampling, padding or truncation is applied.

Versions used for the published data: Python 3.10, mediapipe 0.10.5,
opencv-python 4.8.1.78, numpy 1.26.0 (``pip install -e ".[extract]"``).

Command line::

    python -m signai_landmarks.extraction <videos_dir> <out_dir>

``<videos_dir>`` is expected to contain ``sign_0N/<name>.mp4``; each output is
written to ``<out_dir>/sign_0N/<name>.npy``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

HOLISTIC_PARAMS = {"static_image_mode": False, "model_complexity": 2}


def extract_keypoints(results) -> np.ndarray:
    """Build the (1662,) feature vector for one frame of Holistic results."""
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])


def process_video(video, holistic) -> np.ndarray:
    """Run Holistic on every decoded frame of an open ``cv2.VideoCapture``."""
    import cv2

    processed_points = []
    while True:
        ret, frame = video.read()
        if ret == False:  # noqa: E712 (kept as in the original code)
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)
        processed_points.append(extract_keypoints(results=results))
    return np.array(processed_points)


def transform_video(video) -> np.ndarray:
    """Extract the landmark sequence of one ``cv2.VideoCapture``."""
    import mediapipe as mp

    with mp.solutions.holistic.Holistic(**HOLISTIC_PARAMS) as holistic:
        return process_video(video, holistic)


def extract_file(video_path: str | Path) -> np.ndarray:
    """Extract the landmark sequence of one video file."""
    import cv2

    video = cv2.VideoCapture(str(video_path))
    try:
        return transform_video(video)
    finally:
        video.release()


def extract_directory(videos_dir: str | Path, out_dir: str | Path, pattern: str = "*.mp4") -> int:
    """Extract every video under ``videos_dir`` keeping its relative path.

    Returns the number of files written.
    """
    videos_dir, out_dir = Path(videos_dir), Path(out_dir)
    videos = sorted(videos_dir.rglob(pattern))
    for i, video_path in enumerate(videos, 1):
        target = (out_dir / video_path.relative_to(videos_dir)).with_suffix(".npy")
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, extract_file(video_path))
        print(f"[{i}/{len(videos)}] {target}")
    return len(videos)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract MediaPipe Holistic landmarks as in LSC5.")
    parser.add_argument("videos_dir", help="Folder with sign_0N/<name>.mp4 videos")
    parser.add_argument("out_dir", help="Output folder for sign_0N/<name>.npy arrays")
    parser.add_argument("--pattern", default="*.mp4", help="Glob for video files (default: *.mp4)")
    args = parser.parse_args(argv)
    n = extract_directory(args.videos_dir, args.out_dir, args.pattern)
    print(f"Done: {n} files")


if __name__ == "__main__":
    main()
