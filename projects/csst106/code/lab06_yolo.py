"""Lab 6 - Object Detection with YOLO11: pre-trained inference, custom dataset,
training, validation and testing.

Note: the handout writes the weights as 'YOLOv11n.pt'; the real Ultralytics
file name is 'yolo11n.pt' (no "v"), otherwise the download fails.

Part 2 needs annotated images.  Instead of hand-labelling photos in
LabelImg/Roboflow, this script GENERATES a small custom dataset (circles,
squares, triangles on cluttered backgrounds) and writes the labels in exactly
the YOLO format a labelling tool would export, so the whole pipeline
(prepare -> train -> validate -> infer) runs end to end.
"""
import os
import shutil
import time
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.utils import ASSETS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "site", "img")
WORK = os.path.join(HERE, "yolo_work")
os.makedirs(OUT, exist_ok=True)
os.makedirs(WORK, exist_ok=True)
os.chdir(WORK)                        # weights / runs/ are written here


def save(name):
    plt.savefig(os.path.join(OUT, name), dpi=110, bbox_inches="tight")
    plt.close()


# ===========================================================================
# Part 1: Pre-trained YOLO11n inference (webcam version: lab06_webcam.py)
# ===========================================================================
model = YOLO("yolo11n.pt")
print("pre-trained classes:", len(model.names), "e.g.", list(model.names.values())[:10])
for src in [str(ASSETS / "bus.jpg"), os.path.join(HERE, "data", "my_photo.jpg")]:
    name = os.path.basename(src)
    model(src, verbose=False)                                   # warm-up
    t0 = time.perf_counter()
    r = model(src, conf=0.25, verbose=False)[0]
    ms = (time.perf_counter() - t0) * 1000
    print(f"\n{name}: {len(r.boxes)} objects in {ms:.1f} ms (CPU) -> ~{1000 / ms:.1f} FPS")
    for b in r.boxes:
        print(f"   {model.names[int(b.cls)]:12s} conf={float(b.conf):.2f} box={[int(v) for v in b.xyxy[0]]}")
    cv2.imwrite(os.path.join(OUT, f"lab06_pretrained_{name.split('.')[0]}.jpg"), r.plot())

# ===========================================================================
# Part 2: Prepare a custom dataset in YOLO format
# ===========================================================================
CLASSES = ["circle", "square", "triangle"]
IMG = 320
rng = np.random.default_rng(0)


def random_background():
    bg = rng.integers(40, 220, size=3).astype(np.uint8)
    img = np.full((IMG, IMG, 3), bg, np.uint8)
    for _ in range(rng.integers(5, 15)):                            # clutter: random lines
        p1, p2 = rng.integers(0, IMG, 2), rng.integers(0, IMG, 2)
        cv2.line(img, tuple(int(v) for v in p1), tuple(int(v) for v in p2),
                 tuple(int(v) for v in rng.integers(0, 255, 3)), int(rng.integers(1, 3)))
    noise = rng.normal(0, 8, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def shape_points(cls, cx, cy, s):
    """Polygon outline of one object (a circle is approximated by 48 points)."""
    if cls == 0:
        t = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        pts = np.stack([cx + s * np.cos(t), cy + s * np.sin(t)], 1)
    elif cls == 1:
        pts = cv2.boxPoints(((cx, cy), (1.7 * s, 1.7 * s), rng.uniform(0, 90)))
    else:
        a0 = rng.uniform(0, 2 * np.pi)
        pts = [[cx + s * np.cos(a0 + k * 2 * np.pi / 3), cy + s * np.sin(a0 + k * 2 * np.pi / 3)] for k in range(3)]
    return np.round(pts).astype(np.int32)


def build_dataset(root, splits):
    """Write images/<split>/*.jpg and labels/<split>/*.txt in YOLO format."""
    if os.path.exists(root):
        shutil.rmtree(root)
    for split, n in splits.items():
        os.makedirs(f"{root}/images/{split}"); os.makedirs(f"{root}/labels/{split}")
        for i in range(n):
            img = random_background()
            lines, placed = [], []
            for _ in range(rng.integers(1, 5)):
                s = int(rng.integers(14, 45))
                for _try in range(30):                              # avoid overlapping objects
                    cx, cy = (int(v) for v in rng.integers(s + 2, IMG - s - 2, 2))
                    if all(np.hypot(cx - px, cy - py) > s + ps + 4 for px, py, ps in placed):
                        break
                else:
                    continue
                cls = int(rng.integers(0, 3))
                pts = shape_points(cls, cx, cy, s)
                cv2.fillPoly(img, [pts], tuple(int(v) for v in rng.integers(0, 255, 3)))
                placed.append((cx, cy, s))
                x, y, bw, bh = cv2.boundingRect(pts)               # tight box around the polygon
                # class_id x_center y_center width height  (all normalised to 0..1)
                lines.append(f"{cls} {(x + bw / 2) / IMG:.6f} {(y + bh / 2) / IMG:.6f} {bw / IMG:.6f} {bh / IMG:.6f}")
            cv2.imwrite(f"{root}/images/{split}/{i:04d}.jpg", img)
            with open(f"{root}/labels/{split}/{i:04d}.txt", "w") as f:
                f.write("\n".join(lines) + "\n")


build_dataset("dataset", {"train": 300, "val": 80})
with open("data.yaml", "w") as f:
    f.write(f"path: {os.path.abspath('dataset')}\ntrain: images/train\nval: images/val\n"
            f"nc: {len(CLASSES)}\nnames: {CLASSES}\n")
print(open("data.yaml").read())
print("example label file dataset/labels/train/0000.txt:\n" + open("dataset/labels/train/0000.txt").read())

# Visual check: draw the YOLO labels back onto a few training images
fig, ax = plt.subplots(2, 4, figsize=(16, 8.4))
for a, i in zip(ax.ravel(), range(8)):
    img = cv2.imread(f"dataset/images/train/{i:04d}.jpg")
    for line in open(f"dataset/labels/train/{i:04d}.txt").read().split("\n"):
        if not line.strip():
            continue
        c, xc, yc, bw, bh = line.split(); c = int(c)
        xc, yc, bw, bh = (float(v) * IMG for v in (xc, yc, bw, bh))
        cv2.rectangle(img, (int(xc - bw / 2), int(yc - bh / 2)), (int(xc + bw / 2), int(yc + bh / 2)), (0, 0, 255), 2)
        cv2.putText(img, CLASSES[c], (int(xc - bw / 2), int(yc - bh / 2) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    a.imshow(img[:, :, ::-1]); a.set_title(f"train/{i:04d}.jpg + labels"); a.axis("off")
save("lab06_dataset_samples.png")

# ===========================================================================
# Part 3: Train YOLO11n on the custom dataset (transfer learning from COCO weights)
#   CLI equivalent: yolo task=detect mode=train model=yolo11n.pt data=data.yaml epochs=15 imgsz=320
# ===========================================================================
EPOCHS = int(os.environ.get("YOLO_EPOCHS", 15))
t0 = time.time()
model = YOLO("yolo11n.pt")
model.train(data="data.yaml", epochs=EPOCHS, imgsz=IMG, batch=16, device="cpu", workers=0,
            project=os.path.abspath("runs"), name="train", exist_ok=True, plots=True, seed=0, verbose=False)
print(f"training time: {(time.time() - t0) / 60:.1f} min for {EPOCHS} epochs")

run = str(model.trainer.save_dir)                                  # where Ultralytics actually saved the run
print("run directory:", os.path.relpath(run))
for f in ["results.png", "confusion_matrix.png", "BoxPR_curve.png", "PR_curve.png", "val_batch0_pred.jpg"]:
    if os.path.exists(f"{run}/{f}"):
        shutil.copy(f"{run}/{f}", os.path.join(OUT, f"lab06_{f}"))
print("run files:", sorted(os.listdir(run)))

# ===========================================================================
# Part 4: Evaluate (val) and use the trained model on NEW images
#   CLI equivalent: yolo task=detect mode=val model=runs/train/weights/best.pt data=data.yaml
# ===========================================================================
best = YOLO(f"{run}/weights/best.pt")
metrics = best.val(data="data.yaml", imgsz=IMG, device="cpu", workers=0, plots=False, verbose=False)
print(f"\nVALIDATION  precision={metrics.box.mp:.3f}  recall={metrics.box.mr:.3f}  "
      f"mAP50={metrics.box.map50:.3f}  mAP50-95={metrics.box.map:.3f}")
for i, c in enumerate(CLASSES):
    p, r, ap50, ap = metrics.box.class_result(i)
    print(f"   {c:9s} P={p:.3f} R={r:.3f} AP50={ap50:.3f} AP50-95={ap:.3f}")

# Brand-new test images that were never seen during training or validation
build_dataset("testset", {"test": 8})
fig, ax = plt.subplots(2, 4, figsize=(16, 8.4))
for a, i in zip(ax.ravel(), range(8)):
    res = best(f"testset/images/test/{i:04d}.jpg", conf=0.4, verbose=False)[0]
    a.imshow(res.plot()[:, :, ::-1]); a.set_title(f"new image {i}: {len(res.boxes)} detections"); a.axis("off")
save("lab06_custom_predictions.png")

# Also show the model on a real-world photo where these shapes do not exist (expected: no / few detections)
res = best(str(ASSETS / "bus.jpg"), conf=0.4, verbose=False)[0]
print("custom model on bus.jpg ->", len(res.boxes), "detections (it only knows circle/square/triangle)")
