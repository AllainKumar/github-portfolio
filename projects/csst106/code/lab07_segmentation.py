"""Lab 7 - Segmentation techniques: thresholding, region-based (region growing +
watershed), edge-based (Canny + closed contours) and semantic (DeepLabV3)."""
import os
import time
from collections import deque
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "site", "img")
os.makedirs(OUT, exist_ok=True)


def save(name):
    plt.savefig(os.path.join(OUT, name), dpi=110, bbox_inches="tight")
    plt.close()


def iou(pred, gt):
    pred, gt = pred > 0, gt > 0
    return np.logical_and(pred, gt).sum() / np.logical_or(pred, gt).sum()


def n_objects(mask, min_area=30):
    n, _, st, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8))
    return int(np.sum(st[1:, cv2.CC_STAT_AREA] >= min_area))


coins = cv2.imread(os.path.join(HERE, "data", "coins.png"), cv2.IMREAD_GRAYSCALE)
page = cv2.imread(os.path.join(HERE, "data", "page.png"), cv2.IMREAD_GRAYSCALE)

# ===========================================================================
# Test image WITH ground truth (for IoU): touching discs, lighting gradient, noise
# ===========================================================================
rng = np.random.default_rng(7)
H, W = 300, 400
gt = np.zeros((H, W), np.uint8)
discs = [(60, 70, 32), (118, 70, 28), (210, 80, 35), (330, 75, 30), (80, 200, 30),
         (140, 215, 32), (200, 210, 28), (300, 200, 38), (360, 250, 25), (240, 255, 22)]
for x, y, r in discs:
    cv2.circle(gt, (x, y), r, 255, -1)
ramp = np.tile(np.linspace(40, 175, W), (H, 1))                 # light falls from right to left
synth = ramp + (gt > 0) * 65 + rng.normal(0, 10, (H, W))
synth = np.clip(cv2.GaussianBlur(synth, (3, 3), 0), 0, 255).astype(np.uint8)
print(f"synthetic image: {len(discs)} discs (one touching pair + one chain of three), ground-truth foreground = "
      f"{100 * (gt > 0).mean():.1f}% of pixels")

# ===========================================================================
# 1. Thresholding segmentation
# ===========================================================================
t_otsu, coins_otsu = cv2.threshold(coins, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
t_page, page_otsu = cv2.threshold(page, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
page_ad_mean = cv2.adaptiveThreshold(page, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)
page_ad_gauss = cv2.adaptiveThreshold(page, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
print(f"Otsu threshold coins = {t_otsu:.0f}, page = {t_page:.0f}")

fig, ax = plt.subplots(2, 4, figsize=(20, 9))
ax[0, 0].imshow(coins, cmap="gray"); ax[0, 0].set_title("coins.png")
ax[0, 1].hist(coins.ravel(), bins=256, range=(0, 256), color="gray"); ax[0, 1].axvline(t_otsu, color="r")
ax[0, 1].set_title(f"histogram, Otsu T = {t_otsu:.0f}")
ax[0, 2].imshow(coins_otsu, cmap="gray"); ax[0, 2].set_title("Global Otsu")
coins_ad = cv2.adaptiveThreshold(coins, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 151, -5)
ax[0, 3].imshow(coins_ad, cmap="gray"); ax[0, 3].set_title("Adaptive GAUSSIAN (151x151, C=-5)")
ax[1, 0].imshow(page, cmap="gray"); ax[1, 0].set_title("page.png (uneven illumination)")
ax[1, 1].imshow(page_otsu, cmap="gray"); ax[1, 1].set_title(f"Global Otsu (T = {t_page:.0f})")
ax[1, 2].imshow(page_ad_mean, cmap="gray"); ax[1, 2].set_title("Adaptive MEAN (21x21, C=10)")
ax[1, 3].imshow(page_ad_gauss, cmap="gray"); ax[1, 3].set_title("Adaptive GAUSSIAN (21x21, C=10)")
for a in ax.ravel():
    if a is not ax[0, 1]:
        a.axis("off")
save("lab07_thresholding.png")

# ===========================================================================
# 2. Region-based segmentation
# ===========================================================================
RG_TOLS = (15, 30, 50)


def region_grow(img, seeds, tol):
    """4-connected region growing: add a neighbour while |I - region mean| <= tol."""
    mask = np.zeros(img.shape, np.uint8)
    img_f = img.astype(np.float64)
    for sy, sx in seeds:
        if mask[sy, sx]:
            continue
        total, count = img_f[sy, sx], 1
        q = deque([(sy, sx)]); mask[sy, sx] = 255
        while q:
            y, x = q.popleft()
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < img.shape[0] and 0 <= nx < img.shape[1] and not mask[ny, nx] \
                        and abs(img_f[ny, nx] - total / count) <= tol:
                    mask[ny, nx] = 255; total += img_f[ny, nx]; count += 1; q.append((ny, nx))
    return mask


def clean_mask(binary, min_area=500, max_area=8000):
    """Close gaps, fill the holes inside textured coins, drop specks and the large
    bright background strip (area filter), then smooth with an opening."""
    bw = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    bw = ndi.binary_fill_holes(bw > 0).astype(np.uint8) * 255
    n, lab, st, _ = cv2.connectedComponentsWithStats(bw)
    keep = np.isin(lab, [i for i in range(1, n) if min_area <= st[i, cv2.CC_STAT_AREA] <= max_area])
    return cv2.morphologyEx(keep.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=2)


coins_mask = clean_mask(coins_ad)


def watershed_segment(gray, binary):
    """Marker-based watershed: separates touching objects in a binary mask."""
    k = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=2)
    sure_bg = cv2.dilate(opening, k, iterations=3)
    # pad with background so pixels on the image border get a small distance too
    dist = cv2.distanceTransform(cv2.copyMakeBorder(opening, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0),
                                 cv2.DIST_L2, 5)[1:-1, 1:-1]
    _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    unknown = cv2.subtract(sure_bg, sure_fg)
    n, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1                  # background = 1, objects = 2..n
    markers[unknown == 255] = 0            # 0 = "let the watershed decide"
    markers = cv2.watershed(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), markers)
    return markers, dist, n - 1


coins_markers, coins_dist, coins_n = watershed_segment(coins, coins_mask)
print(f"watershed on coins: {coins_n} objects  (cleaned-mask connected components: {n_objects(coins_mask)}, "
      f"raw Otsu components: {n_objects(coins_otsu)})")

# Seeds: one point inside each coin = the distance-transform peak of each watershed
# region (one seed per coin even where two coins touch; seeds can also be clicked)
seeds = [tuple(int(v) for v in np.unravel_index(np.argmax(np.where(coins_markers == i, coins_dist, 0)), coins.shape))
         for i in range(2, coins_markers.max() + 1)]
coins_blur = cv2.medianBlur(coins, 7)
rg = {tol: region_grow(coins_blur, seeds, tol) for tol in RG_TOLS}
for tol, m in rg.items():
    print(f"region growing tol={tol}: {100 * (m > 0).mean():.1f}% of image labelled")


# Naive watershed (every local minimum of the gradient = a marker) -> over-segmentation
from skimage.segmentation import watershed as sk_watershed
from skimage.filters import sobel
from skimage.feature import peak_local_max
grad = sobel(coins_blur)
minima = peak_local_max(-grad, min_distance=3, exclude_border=False)
naive_markers = np.zeros_like(coins, np.int32)
for i, (y, x) in enumerate(minima, 1):
    naive_markers[y, x] = i
naive = sk_watershed(grad, naive_markers)
print(f"naive gradient watershed: {naive.max()} regions (over-segmentation)")


def color_labels(markers):
    rng_c = np.random.default_rng(1)
    pal = rng_c.integers(40, 255, (markers.max() + 2, 3)).astype(np.uint8)
    pal[1] = 0                                   # background black
    out = pal[np.clip(markers, 0, None)]
    out[markers == -1] = (255, 0, 0)             # watershed boundary
    return out


fig, ax = plt.subplots(2, 4, figsize=(20, 9.5))
ax[0, 0].imshow(coins, cmap="gray"); ax[0, 0].plot([s[1] for s in seeds], [s[0] for s in seeds], "r+", ms=10, mew=2)
ax[0, 0].set_title(f"{len(seeds)} seed points")
for a, tol in zip(ax[0, 1:], RG_TOLS):
    a.imshow(rg[tol], cmap="gray"); a.set_title(f"Region growing, tol = {tol}")
ax[1, 0].imshow(coins_dist, cmap="magma"); ax[1, 0].set_title("Distance transform")
ax[1, 1].imshow(color_labels(coins_markers)); ax[1, 1].set_title(f"Marker watershed: {coins_n} coins")
over = cv2.cvtColor(coins, cv2.COLOR_GRAY2RGB); over[coins_markers == -1] = (255, 0, 0)
ax[1, 2].imshow(over); ax[1, 2].set_title("Watershed boundaries (red)")
ax[1, 3].imshow(naive % 20, cmap="tab20"); ax[1, 3].set_title(f"Naive watershed: {naive.max()} regions (over-seg.)")
for a in ax.ravel():
    a.axis("off")
save("lab07_region.png")

# ===========================================================================
# 3. Edge-based segmentation: Canny -> close gaps -> fill closed contours
# ===========================================================================
def edge_segment(gray, lo, hi, close_iter=2):
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), lo, hi)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=close_iter)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 100]
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    return edges, closed, mask, contours


c_edges, c_closed, c_mask, c_cnts = edge_segment(coins, 50, 150)
print(f"edge-based on coins: {len(c_cnts)} closed contours filled")
noisy = np.clip(coins + rng.normal(0, 20, coins.shape), 0, 255).astype(np.uint8)
n_edges, _, n_mask, n_cnts = edge_segment(noisy, 50, 150)
print(f"edge-based on noisy coins: {len(n_cnts)} contours, {np.count_nonzero(n_edges)} edge px "
      f"vs {np.count_nonzero(c_edges)} clean")

fig, ax = plt.subplots(2, 4, figsize=(20, 9.5))
ax[0, 0].imshow(c_edges, cmap="gray"); ax[0, 0].set_title("Canny edges (50,150)")
ax[0, 1].imshow(c_closed, cmap="gray"); ax[0, 1].set_title("Morphological closing (gap filling)")
ax[0, 2].imshow(c_mask, cmap="gray"); ax[0, 2].set_title(f"Filled closed contours ({len(c_cnts)})")
cmp_img = cv2.cvtColor(coins, cv2.COLOR_GRAY2RGB)
cv2.drawContours(cmp_img, c_cnts, -1, (0, 255, 0), 1)
cmp_img[coins_markers == -1] = (255, 0, 0)
ax[0, 3].imshow(cmp_img); ax[0, 3].set_title("Borders: edge-based (green) vs watershed (red)")
ax[1, 0].imshow(noisy, cmap="gray"); ax[1, 0].set_title("coins + Gaussian noise (sigma=20)")
ax[1, 1].imshow(n_edges, cmap="gray"); ax[1, 1].set_title("Canny on noisy image")
ax[1, 2].imshow(n_mask, cmap="gray"); ax[1, 2].set_title(f"Edge-based result on noise ({len(n_cnts)} contours)")
ax[1, 3].imshow(cv2.Canny(cv2.GaussianBlur(noisy, (5, 5), 0), 120, 250), cmap="gray")
ax[1, 3].set_title("Canny (120,250): fewer false edges, more gaps")
for a in ax.ravel():
    a.axis("off")
save("lab07_edge.png")

# ===========================================================================
# Quantitative comparison on the image with ground truth (IoU)
# ===========================================================================
t_s, s_otsu = cv2.threshold(synth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# block (201) must be much larger than an object, otherwise disc centres turn hollow
s_adapt = cv2.adaptiveThreshold(synth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 201, -10)
s_adapt = cv2.morphologyEx(s_adapt, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
s_seeds = [(y, x) for x, y, r in discs]
s_rg = region_grow(cv2.GaussianBlur(synth, (5, 5), 0), s_seeds, 20)
s_rg_leak = region_grow(cv2.GaussianBlur(synth, (5, 5), 0), s_seeds, 25)       # drift demo
s_filled = ndi.binary_fill_holes(s_adapt > 0).astype(np.uint8) * 255
s_ws_markers, _, s_ws_n = watershed_segment(synth, s_filled)
s_ws = ((s_ws_markers > 1) * 255).astype(np.uint8)
_, _, s_edge, s_cnts = edge_segment(synth, 30, 90, close_iter=3)

rows = []
timings = {}
for name, fn in [("Global Otsu", lambda: cv2.threshold(synth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)),
                 ("Adaptive threshold", lambda: cv2.adaptiveThreshold(synth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 201, -10)),
                 ("Region growing", lambda: region_grow(cv2.GaussianBlur(synth, (5, 5), 0), s_seeds, 20)),
                 ("Watershed", lambda: watershed_segment(synth, s_filled)),
                 ("Edge-based", lambda: edge_segment(synth, 30, 90, close_iter=3))]:
    t0 = time.perf_counter()
    for _ in range(3):
        fn()
    timings[name] = (time.perf_counter() - t0) / 3 * 1000

results = [("Global Otsu", s_otsu, n_objects(s_otsu)), ("Adaptive threshold", s_adapt, n_objects(s_adapt)),
           ("Region growing", s_rg, n_objects(s_rg)), ("Watershed", s_ws, s_ws_n),
           ("Edge-based", s_edge, n_objects(s_edge))]
print(f"\nGround truth: 10 discs. Otsu T on synthetic = {t_s:.0f}")
print(f"{'method':20s} {'IoU':>6s} {'objects found':>14s} {'time (ms)':>10s}")
for name, m, n in results:
    print(f"{name:20s} {iou(m, gt):6.3f} {n:14d} {timings[name]:10.1f}")
print(f"(region growing with tol=25 instead of 20: IoU = {iou(s_rg_leak, gt):.3f} -> leaks, see figure)")

fig, ax = plt.subplots(2, 4, figsize=(20, 8.6))
ax[0, 0].imshow(synth, cmap="gray"); ax[0, 0].set_title("Test image (gradient + noise)")
ax[0, 1].imshow(gt, cmap="gray"); ax[0, 1].set_title("Ground truth (10 discs)")
for a, (name, m, n) in zip(list(ax[0, 2:]) + list(ax[1]), results):
    if name == "Watershed":
        a.imshow(color_labels(s_ws_markers))
    else:
        a.imshow(m, cmap="gray")
    a.set_title(f"{name}\nIoU = {iou(m, gt):.3f}, objects = {n}")
ax[1, 3].imshow(s_rg_leak, cmap="gray")
ax[1, 3].set_title(f"Region growing tol=25 (mean drift -> leak)\nIoU = {iou(s_rg_leak, gt):.3f}")
for a in ax.ravel():
    a.axis("off")
save("lab07_iou.png")

# ===========================================================================
# 4. Semantic segmentation with a pre-trained DeepLabV3 (torchvision)
# ===========================================================================
import torch
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from ultralytics.utils import ASSETS

weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1     # 21 PASCAL-VOC classes
model = deeplabv3_resnet50(weights=weights, progress=False).eval()
classes = weights.meta["categories"]
preprocess = weights.transforms()

# Fixed colour per VOC class (standard PASCAL VOC palette)
def voc_palette(n=21):
    pal = np.zeros((n, 3), np.uint8)
    for i in range(n):
        c, r, g, b = i, 0, 0, 0
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j); g |= ((c >> 1) & 1) << (7 - j); b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        pal[i] = (r, g, b)
    return pal
PAL = voc_palette()

for src in [str(ASSETS / "bus.jpg"), os.path.join(HERE, "data", "my_photo.jpg")]:
    fname = os.path.basename(src)
    bgr = cv2.imread(src)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = preprocess(torch.from_numpy(rgb).permute(2, 0, 1)).unsqueeze(0)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(x)["out"][0]
    ms = (time.perf_counter() - t0) * 1000
    pred = out.argmax(0).numpy().astype(np.uint8)
    pred = cv2.resize(pred, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    present = [(c, 100 * (pred == c).mean()) for c in np.unique(pred)]
    print(f"\nDeepLabV3 on {fname}: {ms:.0f} ms on CPU, input {tuple(x.shape)}")
    for c, pct in present:
        print(f"   {classes[c]:12s} {pct:5.1f}% of pixels")

    color = PAL[pred]
    blend = (0.5 * rgb + 0.5 * color).astype(np.uint8)
    fig, ax = plt.subplots(1, 3, figsize=(18, 6.5))
    ax[0].imshow(rgb); ax[0].set_title(fname)
    ax[1].imshow(color); ax[1].set_title("DeepLabV3 per-pixel class map")
    ax[2].imshow(blend); ax[2].set_title("Overlay")
    handles = [plt.Rectangle((0, 0), 1, 1, color=PAL[c] / 255) for c, _ in present]
    ax[2].legend(handles, [f"{classes[c]} ({p:.0f}%)" for c, p in present], loc="lower right", fontsize=9)
    for a in ax:
        a.axis("off")
    save(f"lab07_semantic_{fname.split('.')[0]}.png")
