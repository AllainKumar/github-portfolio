"""Lab 6 - Part 1: real-time object detection from a webcam with pre-trained YOLO11n.

Run:  python lab06_webcam.py      (press q to quit)
"""
import time
import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")          # downloads the COCO-pretrained weights on first run (80 classes)
cap = cv2.VideoCapture(0)           # 0 = default webcam
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

prev = time.time()
while True:
    ok, frame = cap.read()
    if not ok:
        break
    results = model(frame, conf=0.4, verbose=False)[0]

    # Draw every detection: box + "label confidence"
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id, conf = int(box.cls[0]), float(box.conf[0])
        label = f"{model.names[cls_id]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    now = time.time()
    fps = 1.0 / (now - prev); prev = now
    cv2.putText(frame, f"FPS: {fps:.1f}  objects: {len(results.boxes)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow("YOLO11n real-time detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
