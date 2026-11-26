"""Lab 1 - Image Manipulation: load, explore and manipulate an image with OpenCV."""
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "site", "img")
os.makedirs(OUT, exist_ok=True)
SHOW_WINDOWS = False  # set to True on a desktop to also open cv2.imshow() windows


def show(title, image):
    """Open an OpenCV window (only when SHOW_WINDOWS is enabled)."""
    if SHOW_WINDOWS:
        cv2.imshow(title, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def save(name):
    plt.savefig(os.path.join(OUT, name), dpi=110, bbox_inches="tight")
    plt.close()


def to_rgb(image):
    """OpenCV stores colour images as BGR; Matplotlib expects RGB."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Part 1 - Step 2: Load and explore an image
# ---------------------------------------------------------------------------
img = cv2.imread(os.path.join(HERE, "data", "sample.jpg"))   # loads as BGR uint8
if img is None:
    raise FileNotFoundError("sample.jpg not found")
show("Original (BGR)", img)

height, width, channels = img.shape
print(f"img.shape = {img.shape}  -> height={height}, width={width}, channels={channels}")
print(f"dtype = {img.dtype}, total values = {img.size}, pixels = {height * width}")
print(f"value range = {img.min()} .. {img.max()}")

# BGR ordering: channel 0 = Blue, 1 = Green, 2 = Red
b, g, r = cv2.split(img)
fig, ax = plt.subplots(1, 5, figsize=(17, 3.8))
ax[0].imshow(to_rgb(img)); ax[0].set_title("Correct (BGR -> RGB)")
ax[1].imshow(img); ax[1].set_title("Wrong: BGR shown as RGB")
for a, ch, name in zip(ax[2:], (b, g, r), ("Blue channel [0]", "Green channel [1]", "Red channel [2]")):
    a.imshow(ch, cmap="gray"); a.set_title(name)
for a in ax:
    a.axis("off")
save("lab01_channels.png")

# Convert to grayscale: Y = 0.299 R + 0.587 G + 0.114 B
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
show("Grayscale", gray)
print(f"gray.shape = {gray.shape}  (single channel, {gray.nbytes} bytes vs {img.nbytes} bytes)")

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(to_rgb(img)); ax[0].set_title(f"Original colour {img.shape}")
ax[1].imshow(gray, cmap="gray"); ax[1].set_title(f"Grayscale {gray.shape}")
for a in ax:
    a.axis("off")
save("lab01_original_gray.png")

# ---------------------------------------------------------------------------
# Part 1 - Step 3: Explore image properties (pixel access / modification)
# ---------------------------------------------------------------------------
for (y, x) in [(100, 100), (400, 600), (550, 950), (750, 550)]:   # roof, hair, yellow wall, shirt print
    bb, gg, rr = img[y, x]
    expected = 0.299 * rr + 0.587 * gg + 0.114 * bb
    print(f"pixel (row={y}, col={x}): BGR = ({bb}, {gg}, {rr})  gray = {gray[y, x]}  "
          f"(0.299R+0.587G+0.114B = {expected:.1f})")

modified = img.copy()
y, x = 100, 100
print("before:", modified[y, x])
modified[y, x] = [0, 0, 255]                      # a single pure-red pixel (BGR!)
print("after :", modified[y, x])
modified[160:240, 860:940] = [0, 255, 0]          # an 80x80 green block, visible at full size

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(to_rgb(img)); ax[0].set_title("Before")
ax[1].imshow(to_rgb(modified)); ax[1].set_title("After (pixel (100,100) red + green block)")
zoom = modified[90:111, 90:111]
ax[2].imshow(to_rgb(zoom), interpolation="nearest")
ax[2].set_title("21x21 zoom around (100,100)")
for a in ax:
    a.axis("off")
save("lab01_pixel_edit.png")

# ---------------------------------------------------------------------------
# Part 2 - Step 4: Image manipulation (crop, resize, rotate)
# ---------------------------------------------------------------------------
roi = img[260:580, 470:900]                        # head + phone; rows (y) first, then columns (x)
small = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
large = cv2.resize(img, (int(width * 1.5), int(height * 1.5)), interpolation=cv2.INTER_LINEAR)
rot_cw = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
rot_ccw = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
for name, im in [("ROI crop", roi), ("resized 0.5x", small), ("resized 1.5x", large),
                 ("rotated 90 CW", rot_cw), ("rotated 90 CCW", rot_ccw)]:
    print(f"{name:15s} shape = {im.shape}")

fig, ax = plt.subplots(2, 3, figsize=(15, 10))
items = [(img, f"Original {img.shape[:2]}"), (roi, f"ROI crop {roi.shape[:2]}"),
         (small, f"Resized 0.5x {small.shape[:2]}"), (large, f"Resized 1.5x {large.shape[:2]}"),
         (rot_cw, "Rotated 90 deg clockwise"), (rot_ccw, "Rotated 90 deg counter-clockwise")]
for a, (im, t) in zip(ax.ravel(), items):
    a.imshow(to_rgb(im)); a.set_title(t)
ax[0, 2].set_xlim(0, width); ax[0, 2].set_ylim(height, 0)     # draw on the same scale
ax[1, 0].set_xlim(0, large.shape[1]); ax[1, 0].set_ylim(large.shape[0], 0)
save("lab01_manipulations.png")

# Bonus: an arbitrary-angle rotation needs an affine warp
M = cv2.getRotationMatrix2D((width / 2, height / 2), 30, 1.0)
rot30 = cv2.warpAffine(img, M, (width, height))
cv2.imwrite(os.path.join(OUT, "lab01_rot30.jpg"), rot30)
print("rotation matrix for 30 deg:\n", np.round(M, 3))
