"""Lab 2 - Edge Detection: Sobel vs Canny on input.jpg."""
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "site", "img")
os.makedirs(OUT, exist_ok=True)


def save(name):
    plt.savefig(os.path.join(OUT, name), dpi=110, bbox_inches="tight")
    plt.close()


def edge_detection(image_path):
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Image not found or path is incorrect.")
        return None

    # Apply Gaussian blur to reduce noise
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)

    # Sobel edge detection (detects edges along x and y gradient directions)
    sobelx = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobelx, sobely)

    # Canny edge detection (detects strong edges using gradient and thresholding)
    edges_canny = cv2.Canny(img_blur, 100, 200)

    # Display results side by side (saved to file instead of plt.show())
    plt.figure(figsize=(15, 5.5))
    plt.subplot(1, 3, 1); plt.imshow(img, cmap="gray"); plt.title("Original Image"); plt.axis("off")
    plt.subplot(1, 3, 2); plt.imshow(sobel_combined, cmap="gray"); plt.title("Sobel Edge Detection"); plt.axis("off")
    plt.subplot(1, 3, 3); plt.imshow(edges_canny, cmap="gray"); plt.title("Canny Edge Detection"); plt.axis("off")
    save("lab02_side_by_side.png")
    return img, img_blur, sobelx, sobely, sobel_combined, edges_canny


img, img_blur, sobelx, sobely, sobel, canny = edge_detection(os.path.join(HERE, "data", "input.jpg"))

# ---- Extra analysis used to explain the differences -----------------------
# 1) Sobel directional components: x responds to vertical edges, y to horizontal edges
plt.figure(figsize=(15, 5.5))
for i, (im, t) in enumerate([(np.abs(sobelx), "|Sobel X| (vertical edges)"),
                             (np.abs(sobely), "|Sobel Y| (horizontal edges)"),
                             (sobel, "Magnitude sqrt(Gx^2 + Gy^2)")]):
    plt.subplot(1, 3, i + 1); plt.imshow(im, cmap="gray"); plt.title(t); plt.axis("off")
save("lab02_sobel_xy.png")

# 2) Zoomed crop: thick, graded Sobel response vs thin, binary Canny response
y0, y1, x0, x1 = 130, 290, 230, 450          # head, hand and phone region
plt.figure(figsize=(15, 5.5))
for i, (im, t) in enumerate([(img, "Original (zoom)"), (sobel, "Sobel (zoom)"), (canny, "Canny (zoom)")]):
    plt.subplot(1, 3, i + 1); plt.imshow(im[y0:y1, x0:x1], cmap="gray", interpolation="nearest")
    plt.title(t); plt.axis("off")
save("lab02_zoom.png")

# 3) Effect of the Canny hysteresis thresholds
plt.figure(figsize=(15, 5.5))
for i, (lo, hi) in enumerate([(30, 80), (100, 200), (200, 300)]):
    e = cv2.Canny(img_blur, lo, hi)
    plt.subplot(1, 3, i + 1); plt.imshow(e, cmap="gray")
    plt.title(f"Canny low={lo}, high={hi}\n{100 * np.count_nonzero(e) / e.size:.2f}% edge pixels")
    plt.axis("off")
save("lab02_canny_thresholds.png")

# 4) Numbers
sobel_8u = cv2.convertScaleAbs(sobel)
_, sobel_bin = cv2.threshold(sobel_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"image shape: {img.shape}")
print(f"Sobel magnitude range: {sobel.min():.1f} .. {sobel.max():.1f}, "
      f"unique values: {len(np.unique(sobel_8u))} (continuous / grayscale map)")
print(f"Canny unique values: {np.unique(canny)} (binary map)")
print(f"Sobel pixels above Otsu threshold: {100 * np.count_nonzero(sobel_bin) / sobel_bin.size:.2f}%")
print(f"Canny edge pixels (100,200):       {100 * np.count_nonzero(canny) / canny.size:.2f}%")

# Edge thickness along a horizontal line through the arms, shirt and gym equipment
row = 400
print(f"row {row}: Sobel(Otsu) pixels on = {np.count_nonzero(sobel_bin[row])}, "
      f"Canny pixels on = {np.count_nonzero(canny[row])}")
