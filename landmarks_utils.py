import numpy as np
import face_recognition


_index_lmks_fr = [
    ("left_eyebrow", 0), ("left_eyebrow", 1), ("left_eyebrow", 2), ("left_eyebrow", 3), ("left_eyebrow", 4),
    ("right_eyebrow", 0), ("right_eyebrow", 1), ("right_eyebrow", 2), ("right_eyebrow", 3), ("right_eyebrow", 4),
    ("nose_bridge", 0), ("nose_bridge", 1), ("nose_bridge", 2), ("nose_bridge", 3),
    ("nose_tip", 0), ("nose_tip", 1), ("nose_tip", 2), ("nose_tip", 3), ("nose_tip", 4),
    ("left_eye", 0), ("left_eye", 1), ("left_eye", 2), ("left_eye", 3), ("left_eye", 4), ("left_eye", 5),
    ("right_eye", 0), ("right_eye", 1), ("right_eye", 2), ("right_eye", 3), ("right_eye", 4), ("right_eye", 5),
    ("top_lip", 0), ("top_lip", 1), ("top_lip", 2), ("top_lip", 3), ("top_lip", 4), ("top_lip", 5), ("top_lip", 6),
    ("bottom_lip", 1), ("bottom_lip", 2), ("bottom_lip", 3), ("bottom_lip", 4), ("bottom_lip", 5),
    ("top_lip", 11), ("top_lip", 10), ("top_lip", 9), ("top_lip", 8), ("top_lip", 7),
    ("bottom_lip", 10), ("bottom_lip", 9), ("bottom_lip", 8),
]


def detect_landmarks(img):
    detected = face_recognition.face_landmarks(img)
    if len(detected) == 0:
        return None
    # choose the first face
    detected = detected[0]
    pts = list(map(lambda k: detected[k[0]][k[1]], _index_lmks_fr))
    return np.reshape(np.asarray(pts, dtype=np.float64), (-1, 2))
