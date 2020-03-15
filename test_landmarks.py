import cv2
import numpy as np
from landmarks_utils import detect_landmarks

img = cv2.imread("data/imgHQ00039.jpeg")
pts_true = np.load("data/imgHQ00039_lmks.npy")
pts_pred = detect_landmarks(img)


def put_text(img, text, pos, color, scale=1):
    return cv2.putText(
        img, text, pos,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=scale,
        color=color,
    )


red = (0, 0, 255)
green = (0, 255, 0)
for i, p in enumerate(pts_true):
    x, y = p
    x = int(np.round(x))
    y = int(np.round(y))
    cv2.circle(img, (x, y), 3, green, 1)
    put_text(img, str(i), (x, y-5), green, scale=0.5)


for i, p in enumerate(pts_pred):
    x, y = p
    x = int(np.round(x))
    y = int(np.round(y))
    cv2.circle(img, (x, y), 2, red, 2)
    put_text(img, str(i), (x, y+15), red, scale=0.5)


cv2.imshow("img", img)
cv2.waitKey()
