from types import SimpleNamespace

import numpy as np

from signai_landmarks.constants import FEATURE_DIM
from signai_landmarks.extraction import HOLISTIC_PARAMS, extract_keypoints
from signai_landmarks.keypoints import split_frame


def _component(n_points, start, with_visibility=False):
    points = []
    for i in range(n_points):
        v = start + i
        kwargs = {"x": v + 0.1, "y": v + 0.2, "z": v + 0.3}
        if with_visibility:
            kwargs["visibility"] = v + 0.4
        points.append(SimpleNamespace(**kwargs))
    return SimpleNamespace(landmark=points)


def _results(pose=True, face=True, lh=True, rh=True):
    return SimpleNamespace(
        pose_landmarks=_component(33, 0, with_visibility=True) if pose else None,
        face_landmarks=_component(468, 100) if face else None,
        left_hand_landmarks=_component(21, 1000) if lh else None,
        right_hand_landmarks=_component(21, 2000) if rh else None,
    )


def test_holistic_parameters_match_methods():
    assert HOLISTIC_PARAMS == {"static_image_mode": False, "model_complexity": 2}


def test_vector_layout_and_dtype():
    vec = extract_keypoints(_results())
    assert vec.shape == (FEATURE_DIM,)
    assert vec.dtype == np.float64

    parts = split_frame(vec)
    assert np.allclose(parts["pose"][0], [0.1, 0.2, 0.3, 0.4])
    assert np.allclose(parts["face"][0], [100.1, 100.2, 100.3])
    assert np.allclose(parts["left_hand"][0], [1000.1, 1000.2, 1000.3])
    assert np.allclose(parts["right_hand"][20], [2020.1, 2020.2, 2020.3])


def test_missing_components_are_zero_blocks():
    vec = extract_keypoints(_results(face=False, lh=False))
    parts = split_frame(vec)
    assert vec.shape == (FEATURE_DIM,)
    assert not parts["face"].any()
    assert not parts["left_hand"].any()
    assert parts["pose"].all()
    assert parts["right_hand"].all()


def test_no_detection_gives_all_zero_frame():
    vec = extract_keypoints(_results(False, False, False, False))
    assert vec.shape == (FEATURE_DIM,)
    assert not vec.any()
