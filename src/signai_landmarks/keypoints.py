import numpy as np

POSE_LANDMARKS, POSE_DIMS = 33, 4
FACE_LANDMARKS, FACE_DIMS = 468, 3
HAND_LANDMARKS, HAND_DIMS = 21, 3

POSE_START = 0
POSE_END = POSE_START + POSE_LANDMARKS * POSE_DIMS
FACE_END = POSE_END + FACE_LANDMARKS * FACE_DIMS
LEFT_HAND_END = FACE_END + HAND_LANDMARKS * HAND_DIMS
RIGHT_HAND_END = LEFT_HAND_END + HAND_LANDMARKS * HAND_DIMS

# MediaPipe Holistic's standard skeleton topology, used to draw connecting
# lines between landmarks that belong to the same limb/finger.
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def split_frame(frame: np.ndarray) -> dict[str, np.ndarray]:
    """Split one (1662,) landmark frame into its MediaPipe Holistic parts."""
    return {
        "pose": frame[POSE_START:POSE_END].reshape(POSE_LANDMARKS, POSE_DIMS),
        "face": frame[POSE_END:FACE_END].reshape(FACE_LANDMARKS, FACE_DIMS),
        "left_hand": frame[FACE_END:LEFT_HAND_END].reshape(HAND_LANDMARKS, HAND_DIMS),
        "right_hand": frame[LEFT_HAND_END:RIGHT_HAND_END].reshape(HAND_LANDMARKS, HAND_DIMS),
    }
