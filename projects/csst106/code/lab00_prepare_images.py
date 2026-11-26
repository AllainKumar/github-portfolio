"""Prepare the images used by the labs in labs/data/.

Labs 1-4 use our own photo, data/my_photo.jpg (1080 x 1080 colour gym photo).
Labs 2-4 get a 540 x 540 copy so that thin edges and keypoints stay visible in
the figures and the runtimes stay short.  Lab 7's classical segmentation uses
scikit-image's public-domain "coins" and "page" images, because those methods
need simple objects on a background (and the IoU study uses a synthetic image).
"""
import os
import cv2
import numpy as np
from skimage import data

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)

photo = cv2.imread(os.path.join(DATA, "my_photo.jpg"))
if photo is None:
    raise FileNotFoundError("put your photo in labs/data/my_photo.jpg")
small = cv2.resize(photo, (540, 540), interpolation=cv2.INTER_AREA)

cv2.imwrite(os.path.join(DATA, "sample.jpg"), photo)      # Lab 1 - full-size colour photo
cv2.imwrite(os.path.join(DATA, "input.jpg"), small)       # Lab 2 - "input.jpg" (read as grayscale)
#                                                           Lab 3 - also reads input.jpg as grayscale
cv2.imwrite(os.path.join(DATA, "scene.jpg"), small)       # Lab 4 - feature detection

# Lab 3 - simulate a hazy / washed-out colour photo (compressed intensity range)
hazy = np.clip(small.astype(np.float32) * 0.35 + 95, 0, 255).astype(np.uint8)
cv2.imwrite(os.path.join(DATA, "hazy_color.jpg"), hazy)

cv2.imwrite(os.path.join(DATA, "coins.png"), data.coins())  # Lab 7 - objects on background
cv2.imwrite(os.path.join(DATA, "page.png"), data.page())    # Lab 7 - uneven illumination

print("Saved:", sorted(os.listdir(DATA)))
