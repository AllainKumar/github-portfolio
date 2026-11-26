"""Lab 4 - SURF feature detection, matching, comparison with SIFT/ORB, and U-SURF.

SURF is patented, so the opencv-contrib-python wheels on PyPI are built WITHOUT
the "nonfree" module and cv2.xfeatures2d.SURF_create() raises an error.
This script uses real SURF when it is available (OpenCV built with
OPENCV_ENABLE_NONFREE=ON) and otherwise:
  * reproduces SURF's detector stage (Fast-Hessian = determinant of the Hessian
    computed with integral images / box filters) with skimage.feature.blob_doh
    to study the Hessian threshold, and
  * demonstrates the U-SURF idea (skip orientation assignment) with upright SIFT.
"""
import os
import time
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


# ---------------------------------------------------------------------------
# 1. Import libraries and load image
# ---------------------------------------------------------------------------
img = cv2.imread(os.path.join(HERE, "data", "scene.jpg"), cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Error loading image")
h, w = img.shape

# Second view of the same scene: rotated 30 deg, scaled 0.8x, brighter.
# Because we create it ourselves the TRUE transform M is known, so every match
# can be checked automatically (ground truth for matching accuracy).
def make_view(angle, scale, gain=1.15, bias=10):
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    view = cv2.warpAffine(img, M, (w, h), borderValue=0)
    view = cv2.convertScaleAbs(view, alpha=gain, beta=bias)
    return view, M

img2, M_true = make_view(30, 0.8)

# ---------------------------------------------------------------------------
# 2. Create SURF detector (if this OpenCV build has it)
# ---------------------------------------------------------------------------
try:
    cv2.xfeatures2d.SURF_create(400)
    SURF_AVAILABLE = True
except (cv2.error, AttributeError) as e:
    SURF_AVAILABLE = False
    print("SURF not available in this OpenCV build:", str(e).strip().splitlines()[-1])
print("OpenCV", cv2.__version__, "| SURF available:", SURF_AVAILABLE)

if SURF_AVAILABLE:
    hessian_threshold = 400
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessian_threshold)
    keypoints, descriptors = surf.detectAndCompute(img, None)
    print(f"Number of keypoints detected: {len(keypoints)}")

    # 3. Draw keypoints with size and orientation
    img_kp = cv2.drawKeypoints(img, keypoints, None, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(os.path.join(OUT, "lab04_surf_keypoints.png"), img_kp)

    # Q1: Hessian threshold sweep with real SURF
    thresholds = [100, 400, 1000, 2000, 5000]
    counts = [len(cv2.xfeatures2d.SURF_create(t).detect(img, None)) for t in thresholds]
else:
    # Q1 with SURF's detector stage reproduced by blob_doh (determinant of Hessian,
    # integral-image box filters).  Its threshold plays the role of hessianThreshold.
    from skimage.feature import blob_doh
    norm = img.astype(np.float64) / 255.0
    # overlap=1.0 disables blob pruning, so (like SURF) only non-maximum suppression is applied
    thresholds = [0.0005, 0.001, 0.002, 0.005, 0.01]
    blobs_by_t = {t: blob_doh(norm, min_sigma=2, max_sigma=30, num_sigma=10, threshold=t, overlap=1.0)
                  for t in thresholds}
    counts = [len(blobs_by_t[t]) for t in thresholds]

    fig, ax = plt.subplots(1, 3, figsize=(18, 6.2))
    for a, t in zip(ax, [thresholds[1], thresholds[2], thresholds[3]]):
        a.imshow(img, cmap="gray")
        for y, x, s in blobs_by_t[t]:
            a.add_patch(plt.Circle((x, y), s * 1.414, color="lime", fill=False, lw=0.9))
        a.set_title(f"Fast-Hessian (det of Hessian), threshold={t}\n{len(blobs_by_t[t])} keypoints")
        a.axis("off")
    save("lab04_hessian_blobs.png")

for t, c in zip(thresholds, counts):
    print(f"Hessian threshold {t:>8}: {c} keypoints")
plt.figure(figsize=(7, 4.2))
plt.plot([str(t) for t in thresholds], counts, "o-", color="crimson")
plt.xlabel("Hessian threshold"); plt.ylabel("number of keypoints"); plt.yscale("log"); plt.grid(alpha=0.3)
plt.title("Higher Hessian threshold -> fewer, stronger keypoints")
save("lab04_threshold_curve.png")

# ---------------------------------------------------------------------------
# LAB ACTIVITY: matching + SURF vs SIFT vs ORB
# ---------------------------------------------------------------------------
detectors = {}
if SURF_AVAILABLE:
    detectors["SURF"] = (cv2.xfeatures2d.SURF_create(400), cv2.NORM_L2)
    detectors["U-SURF"] = (cv2.xfeatures2d.SURF_create(400, upright=True), cv2.NORM_L2)
detectors["SIFT"] = (cv2.SIFT_create(), cv2.NORM_L2)
detectors["ORB"] = (cv2.ORB_create(nfeatures=2000), cv2.NORM_HAMMING)


def is_correct(pt1, pt2, M, tol=4.0):
    """A match is correct if the true transform maps pt1 within tol pixels of pt2."""
    p = M @ np.array([pt1[0], pt1[1], 1.0])
    return np.hypot(p[0] - pt2[0], p[1] - pt2[1]) <= tol


def match(name, det, norm, a, b, M, draw=None):
    t0 = time.perf_counter()
    kp1, d1 = det.detectAndCompute(a, None)
    kp2, d2 = det.detectAndCompute(b, None)
    t_detect = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    knn = cv2.BFMatcher(norm).knnMatch(d1, d2, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < 0.75 * n.distance]   # Lowe ratio test
    t_match = (time.perf_counter() - t0) * 1000
    correct = sum(is_correct(kp1[m.queryIdx].pt, kp2[m.trainIdx].pt, M) for m in good)
    res = dict(kp1=len(kp1), kp2=len(kp2), dim=d1.shape[1], dtype=str(d1.dtype), good=len(good),
               correct=correct, precision=correct / max(len(good), 1),
               t_detect=t_detect, t_match=t_match)
    if draw:
        good_sorted = sorted(good, key=lambda m: m.distance)[:60]
        vis = cv2.drawMatches(a, kp1, b, kp2, good_sorted, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.figure(figsize=(14, 6.5)); plt.imshow(vis[:, :, ::-1])
        plt.title(f"{name}: best 60 of {len(good)} ratio-test matches "
                  f"({correct} geometrically correct, {100 * res['precision']:.1f}%)")
        plt.axis("off"); save(draw)
    return res, kp1


results = {}
for name, (det, norm) in detectors.items():
    det.detectAndCompute(img, None)                                   # warm-up for fair timing
    r, kp = match(name, det, norm, img, img2, M_true, draw=f"lab04_matches_{name.lower()}.png")
    results[name] = r
    print(f"{name:7s} kp={r['kp1']:5d}/{r['kp2']:5d} desc={r['dim']:3d}-D {r['dtype']:7s} "
          f"good={r['good']:5d} correct={r['correct']:5d} ({100 * r['precision']:.1f}%) "
          f"detect+describe(2 imgs)={r['t_detect']:.1f} ms  match={r['t_match']:.1f} ms")

# Keypoint visualisation for each detector (rich keypoints: size + orientation)
fig, ax = plt.subplots(1, len(detectors), figsize=(6 * len(detectors), 6.2))
for a, (name, (det, _)) in zip(np.atleast_1d(ax), detectors.items()):
    kps = det.detect(img, None)
    vis = cv2.drawKeypoints(img, kps, None, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    a.imshow(vis[:, :, ::-1]); a.set_title(f"{name}: {len(kps)} keypoints"); a.axis("off")
save("lab04_keypoints_compare.png")

# ---------------------------------------------------------------------------
# U-SURF / upright mode: skip orientation assignment
# ---------------------------------------------------------------------------
def upright_sift_match(a, b, M):
    """Upright variant of SIFT: force every keypoint angle to 0 before describing."""
    sift = cv2.SIFT_create()
    def upright(kps):
        # SIFT returns one keypoint per dominant orientation; with angle=0 those copies
        # become identical and would break the ratio test, so keep one per location/scale.
        unique = {}
        for k in kps:
            unique.setdefault((round(k.pt[0], 1), round(k.pt[1], 1), round(k.size, 1)), k)
        for k in unique.values():
            k.angle = 0
        return list(unique.values())

    kp1 = upright(sift.detect(a, None)); kp2 = upright(sift.detect(b, None))
    kp1, d1 = sift.compute(a, kp1); kp2, d2 = sift.compute(b, kp2)
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < 0.75 * n.distance]
    correct = sum(is_correct(kp1[m.queryIdx].pt, kp2[m.trainIdx].pt, M) for m in good)
    return correct, len(good)

print("\nUpright (orientation-free) vs rotation-invariant descriptors:")
rows = []
for angle in [0, 5, 10, 20, 45, 90]:
    view, M = make_view(angle, 1.0)
    if SURF_AVAILABLE:
        rc, _ = match("SURF", cv2.xfeatures2d.SURF_create(400), cv2.NORM_L2, img, view, M)
        uc, _ = match("U-SURF", cv2.xfeatures2d.SURF_create(400, upright=True), cv2.NORM_L2, img, view, M)
        rot_c, up_c = rc["correct"], uc["correct"]
        labels = ("SURF", "U-SURF")
    else:
        rc, _ = match("SIFT", cv2.SIFT_create(), cv2.NORM_L2, img, view, M)
        rot_c = rc["correct"]
        up_c, _ = upright_sift_match(img, view, M)
        labels = ("SIFT (oriented)", "Upright SIFT (U-SURF idea)")
    rows.append((angle, rot_c, up_c))
    print(f"rotation {angle:3d} deg: {labels[0]} correct={rot_c:5d}   {labels[1]} correct={up_c:5d}")

plt.figure(figsize=(7.5, 4.3))
plt.plot([r[0] for r in rows], [r[1] for r in rows], "o-", label=labels[0])
plt.plot([r[0] for r in rows], [r[2] for r in rows], "s-", label=labels[1])
plt.xlabel("in-plane rotation of second image (deg)"); plt.ylabel("correct matches")
plt.title("Upright descriptors only work when the camera is not rotated"); plt.grid(alpha=0.3); plt.legend()
save("lab04_upright.png")
