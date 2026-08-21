import numpy as np

from signai_landmarks.constants import FEATURE_DIM
from signai_landmarks.keypoints import split_frame


def test_split_frame_shapes_and_full_coverage():
    frame = np.arange(FEATURE_DIM, dtype=float)

    parts = split_frame(frame)

    assert parts["pose"].shape == (33, 4)
    assert parts["face"].shape == (468, 3)
    assert parts["left_hand"].shape == (21, 3)
    assert parts["right_hand"].shape == (21, 3)

    reassembled = np.concatenate([part.flatten() for part in parts.values()])
    assert reassembled.shape == (FEATURE_DIM,)
    assert np.array_equal(reassembled, frame)
