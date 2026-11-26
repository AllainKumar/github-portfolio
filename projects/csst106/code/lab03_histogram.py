"""Lab 3 - Histogram Equalization and Contrast Adjustment."""
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


def hist(gray):
    """256-bin histogram of an 8-bit grayscale image."""
    return cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()


def entropy(gray):
    p = hist(gray) / gray.size
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def stats(name, gray):
    print(f"{name:28s} min={gray.min():3d} max={gray.max():3d} mean={gray.mean():6.1f} "
          f"std(RMS contrast)={gray.std():5.1f} entropy={entropy(gray):.2f} bits "
          f"levels used={np.count_nonzero(hist(gray))}")


def show_with_hist(images, titles, name):
    """Images on the top row, their histograms (+ normalised CDF) underneath."""
    n = len(images)
    fig, ax = plt.subplots(2, n, figsize=(4.6 * n, 8))
    for i, (im, t) in enumerate(zip(images, titles)):
        ax[0, i].imshow(im, cmap="gray", vmin=0, vmax=255); ax[0, i].set_title(t); ax[0, i].axis("off")
        h = hist(im)
        ax[1, i].bar(range(256), h, width=1.0, color="steelblue")
        cdf = h.cumsum()
        ax2 = ax[1, i].twinx()
        ax2.plot(cdf / cdf[-1], color="crimson", lw=1.3); ax2.set_ylim(0, 1.05)
        ax[1, i].set_xlim(0, 255); ax[1, i].set_xlabel("intensity")
        ax[1, i].set_title("histogram (bars) + CDF (red)")
    plt.tight_layout()
    save(name)


# ===========================================================================
# Part 1: Histogram equalization
# ===========================================================================
# 1-2. Load the image and convert it to grayscale ---------------------------
gray = cv2.imread(os.path.join(HERE, "data", "input.jpg"), cv2.IMREAD_GRAYSCALE)

# 3. Histogram of the input ------------------------------------------------
h_orig = hist(gray)

# 4a. Global histogram equalization for grayscale ---------------------------
eq = cv2.equalizeHist(gray)

# 5. Display original vs equalized and both histograms ---------------------
show_with_hist([gray, eq], ["Original (dark, underexposed)", "cv2.equalizeHist()"], "lab03_part1_gray.png")

# 4b. Colour image: equalize ONLY the luminance channel ---------------------
color = cv2.imread(os.path.join(HERE, "data", "hazy_color.jpg"))
yuv = cv2.cvtColor(color, cv2.COLOR_BGR2YUV)
yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
eq_yuv = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
eq_hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# Wrong way, for comparison: equalize B, G and R independently
eq_rgb_naive = cv2.merge([cv2.equalizeHist(c) for c in cv2.split(color)])

fig, ax = plt.subplots(1, 4, figsize=(20, 4.4))
for a, (im, t) in zip(ax, [(color, "Original (hazy colour)"), (eq_yuv, "Equalized Y of YUV"),
                           (eq_hsv, "Equalized V of HSV"), (eq_rgb_naive, "Naive: each B,G,R separately")]):
    a.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); a.set_title(t); a.axis("off")
save("lab03_part1_color.png")

# ===========================================================================
# Part 2: Contrast adjustment
# ===========================================================================
# 1. Linear contrast stretching: map [r_min, r_max] -> [0, 255]
def contrast_stretch(img, low_pct=0, high_pct=100):
    lo, hi = np.percentile(img, (low_pct, high_pct))
    out = (img.astype(np.float32) - lo) * 255.0 / (hi - lo)
    return np.clip(out, 0, 255).astype(np.uint8)

stretched = contrast_stretch(gray)              # pure min-max
stretched_p = contrast_stretch(gray, 2, 98)     # ignores 2 % outliers at each end
alpha_beta = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)     # g(x) = alpha*f(x) + beta

# 2. CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl = clahe.apply(gray)

# 3. Visualise: original, linear stretch, global HE, CLAHE + histograms
show_with_hist([gray, stretched_p, eq, cl],
               ["Original", "Linear stretch (2-98 %)", "Global HE", "CLAHE (clip=2, 8x8)"],
               "lab03_part2_compare.png")

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for a, (im, t) in zip(ax, [(stretched, "Min-max stretch (0-100 %)"), (stretched_p, "Percentile stretch (2-98 %)"),
                           (alpha_beta, "alpha=1.5, beta=30")]):
    a.imshow(im, cmap="gray", vmin=0, vmax=255); a.set_title(t); a.axis("off")
save("lab03_linear_variants.png")

# CLAHE parameter study
fig, ax = plt.subplots(1, 4, figsize=(20, 5))
for a, (clip, tile) in zip(ax, [(1.0, 8), (2.0, 8), (4.0, 8), (2.0, 16)]):
    out = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(gray)
    a.imshow(out, cmap="gray"); a.set_title(f"clipLimit={clip}, tiles={tile}x{tile}"); a.axis("off")
save("lab03_clahe_params.png")

# ===========================================================================
# Part 3: Histogram equalization written from scratch
# ===========================================================================
def equalize_manual(img):
    """T(r) = round((CDF(r) - CDF_min) / (N - CDF_min) * 255)  -- same rule OpenCV uses."""
    h = np.bincount(img.ravel(), minlength=256)
    cdf = h.cumsum()
    cdf_min = cdf[np.nonzero(cdf)[0][0]]
    lut = np.round((cdf - cdf_min) / (img.size - cdf_min) * 255)
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return lut[img], lut

eq_manual, lut = equalize_manual(gray)
diff = np.abs(eq_manual.astype(int) - eq.astype(int))
print(f"manual vs cv2.equalizeHist: max abs difference = {diff.max()}, "
      f"pixels differing = {np.count_nonzero(diff)}")

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
ax[0].plot(lut, color="crimson"); ax[0].plot([0, 255], [0, 255], "k--", lw=0.8)
ax[0].set_title("Mapping T(r) learned from the CDF"); ax[0].set_xlabel("input r"); ax[0].set_ylabel("output s")
ax[1].imshow(eq_manual, cmap="gray"); ax[1].set_title("Manual equalization"); ax[1].axis("off")
ax[2].imshow(diff, cmap="gray", vmin=0, vmax=1); ax[2].set_title(f"|manual - OpenCV| (max={diff.max()})"); ax[2].axis("off")
save("lab03_part3_manual.png")

# ===========================================================================
# Part 4: Observations - numbers
# ===========================================================================
for name, im in [("original", gray), ("min-max stretch", stretched), ("percentile stretch", stretched_p),
                 ("global HE", eq), ("CLAHE", cl)]:
    stats(name, im)

# Noise amplification: std inside a visually flat 40x40 patch of the black T-shirt
patch = (slice(460, 500), slice(290, 330))
for name, im in [("original", gray), ("global HE", eq), ("CLAHE", cl)]:
    print(f"flat-patch std ({name:9s}) = {im[patch].astype(float).std():.2f}")

# Zoom on the flat patch area to make the amplification visible
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for a, (im, t) in zip(ax, [(gray, "Original"), (eq, "Global HE"), (cl, "CLAHE")]):
    a.imshow(im[420:540, 250:370], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    a.set_title(t + " - 120x120 crop of the black shirt"); a.axis("off")
save("lab03_noise_zoom.png")
