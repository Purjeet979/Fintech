from ultralytics import YOLO
from pathlib import Path
from PIL import Image

# -----------------------
# USER INPUTS
# -----------------------
MODEL_PATH = "best.pt"  # change if needed
IMAGE_PATH = "90018918722_175278173_1_pg22.png"                  # change if needed
CONF_THRESH = 0.25

# -----------------------
# Load model and image
# -----------------------
model = YOLO(MODEL_PATH)

img = Image.open(IMAGE_PATH)
W, H = img.size

# -----------------------
# Run inference
# -----------------------
results = model.predict(
    source=IMAGE_PATH,
    conf=CONF_THRESH,
    verbose=False
)

# -----------------------
# Print YOLO-format boxes
# -----------------------
print("YOLO format: <class_id> <x_center> <y_center> <width> <height>\n")

for r in results:
    if r.boxes is None:
        continue

    boxes = r.boxes
    for i in range(len(boxes)):
        b = boxes[i]

        # class id
        cls = int(b.cls[0]) if hasattr(b.cls, "__len__") else int(b.cls)

        # xyxy in pixels
        x1, y1, x2, y2 = b.xyxy[0].tolist()

        # convert to YOLO normalized format
        x_center = ((x1 + x2) / 2) / W
        y_center = ((y1 + y2) / 2) / H
        width    = (x2 - x1) / W
        height   = (y2 - y1) / H

        print(
            f"{cls} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )
