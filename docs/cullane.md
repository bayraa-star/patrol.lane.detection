# CULane Support

This project can use CULane by converting its native `.lines` annotations into a YOLO dataset.

## Convert CULane

```bash
python3 scripts/cullane_to_yolo.py \
  --culane-root /path/to/CULane \
  --output-root /path/to/cullane-yolo \
  --task segment
```

The converter writes:

- `/path/to/cullane-yolo/images/{train,val,test}/...`
- `/path/to/cullane-yolo/labels/{train,val,test}/...`
- `/path/to/cullane-yolo/cullane.yaml`

By default images are symlinked. Use `--image-mode copy` if your training environment cannot follow symlinks.

CULane stores lane markings as polylines, not filled masks. For `--task segment`, the converter expands each polyline into a thin polygon using `--lane-width 16`. Use `--task detect` if you want bounding-box labels instead.

## Train

Example Ultralytics segmentation training:

```bash
yolo segment train \
  model=yolov8n-seg.pt \
  data=/path/to/cullane-yolo/cullane.yaml \
  imgsz=1280 \
  epochs=100
```

Example detection training:

```bash
yolo detect train \
  model=yolov8n.pt \
  data=/path/to/cullane-yolo/cullane.yaml \
  imgsz=1280 \
  epochs=100
```

## Run This Service With A CULane Model

Point the service at the trained weights and generated CULane config:

```bash
MODEL_PATH=/path/to/best.pt \
MODEL_CONFIG_PATH=/path/to/cullane-yolo/cullane.yaml \
MODEL_BACKEND=ultralytics \
python main.py
```

For OpenVINO, export the trained model first:

```bash
python3 conver_into_onnx.py \
  --model /path/to/best.pt \
  --output models/lane/cullane.onnx \
  --export-openvino \
  --openvino-dir models/lane/cullane_openvino
```

Then run with:

```bash
MODEL_PATH=models/lane/cullane_openvino/cullane.xml \
MODEL_CONFIG_PATH=/path/to/cullane-yolo/cullane.yaml \
MODEL_BACKEND=openvino \
python main.py
```
