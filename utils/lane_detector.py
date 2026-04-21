import logging
import os
from pathlib import Path
import subprocess
import threading
import time
from queue import Empty, Queue

import cv2
import numpy as np
from ultralytics import YOLO


class FrameBuffer:
    def __init__(self, maxsize=1):
        self.queue = Queue(maxsize=maxsize)

    def put(self, item):
        if not self.queue.full():
            self.queue.put(item)
            return

        try:
            self.queue.get_nowait()
        except Empty:
            pass

        try:
            self.queue.put(item, block=False)
        except Exception:
            pass

    def get(self, timeout=None):
        try:
            if timeout is None:
                return self.queue.get_nowait()
            return self.queue.get(timeout=timeout)
        except Empty:
            return None


def get_roi_coordinates(frame_shape, roi_ratios):
    height, width = frame_shape[:2]
    return (
        int(roi_ratios[0] * width),
        int(roi_ratios[1] * height),
        int(roi_ratios[2] * width),
        int(roi_ratios[3] * height),
    )


def is_bbox_in_roi(bbox, roi_coords):
    center_x = int((bbox[0] + bbox[2]) / 2)
    center_y = int((bbox[1] + bbox[3]) / 2)
    return roi_coords[0] <= center_x <= roi_coords[2] and roi_coords[1] <= center_y <= roi_coords[3]


def draw_roi(image, roi_coords, color, thickness):
    cv2.rectangle(
        image,
        (roi_coords[0], roi_coords[1]),
        (roi_coords[2], roi_coords[3]),
        color,
        thickness,
    )


def draw_detections(
    image,
    detections,
    class_names,
    class_colors,
    roi_coords=None,
    line_thickness=2,
    text_scale=0.6,
    text_thickness=2,
):
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        class_id = detection["class_id"]
        confidence = detection["confidence"]
        polygon = detection.get("polygon")
        render_as = detection.get("render_as", "bbox")

        color = class_colors[class_id % len(class_colors)] if class_colors else (0, 255, 0)
        class_name = class_names[class_id] if 0 <= class_id < len(class_names) else f"class_{class_id}"
        label = f"{class_name} {confidence:.2f}"
        if roi_coords is not None and is_bbox_in_roi(detection["bbox"], roi_coords):
            label += " [ROI]"

        if render_as == "polygon" and polygon:
            points = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
            overlay = image.copy()
            cv2.fillPoly(overlay, [points], color)
            cv2.addWeighted(overlay, 0.20, image, 0.80, 0.0, image)
            cv2.polylines(image, [points], True, color, line_thickness, cv2.LINE_AA)
        else:
            cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)

        (label_width, label_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            text_thickness,
        )
        top = max(0, y1 - label_height - baseline - 6)
        cv2.rectangle(
            image,
            (x1, top),
            (x1 + label_width + 8, y1),
            color,
            thickness=-1,
        )
        cv2.putText(
            image,
            label,
            (x1 + 4, y1 - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            (0, 0, 0),
            text_thickness,
            cv2.LINE_AA,
        )


class RTSPConnection:
    def __init__(self, rtsp_url, max_retries=10, retry_delay=2, connection_timeout=15, config=None, fps=None):
        self.rtsp_url = rtsp_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.connection_timeout = connection_timeout
        self.cap = None
        self.is_connected = False
        self.config = config
        self.fps = fps
        self.active_decoder = None
        self.active_decode_mode = None
        self.active_drm_device = None
        self.active_libva_driver = None

    @staticmethod
    def _has_gstreamer_element(element_name):
        try:
            result = subprocess.run(
                ["gst-inspect-1.0", element_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except FileNotFoundError:
            return False
        return result.returncode == 0

    @staticmethod
    def _read_text(path):
        try:
            return Path(path).read_text(encoding="utf-8").strip()
        except OSError:
            return None

    def _is_intel_render_device(self, drm_device):
        device_name = Path(drm_device).name
        vendor_path = Path("/sys/class/drm") / device_name / "device" / "vendor"
        vendor_id = self._read_text(vendor_path)
        return vendor_id == "0x8086"

    def _discover_intel_drm_devices(self):
        configured = getattr(self.config, "INTEL_DRM_DEVICE", "")
        auto_detect = getattr(self.config, "AUTO_DETECT_INTEL_DRM", True)
        candidates = []

        if configured:
            if os.path.exists(configured):
                if self._is_intel_render_device(configured):
                    candidates.append(configured)
                else:
                    logging.warning("Configured DRM device '%s' is not Intel; ignoring it", configured)
            else:
                logging.warning("Configured DRM device '%s' does not exist; ignoring it", configured)

        if not auto_detect:
            return candidates

        for entry in sorted(Path("/sys/class/drm").glob("renderD*")):
            vendor_id = self._read_text(entry / "device" / "vendor")
            if vendor_id != "0x8086":
                continue
            drm_device = Path("/dev/dri") / entry.name
            if drm_device.exists() and str(drm_device) not in candidates:
                candidates.append(str(drm_device))

        return candidates

    @staticmethod
    def _reset_intel_decode_env():
        os.environ.pop("GST_VAAPI_DRM_DEVICE", None)
        os.environ.pop("LIBVA_DRIVER_NAME", None)

    def _configure_intel_decode(self, drm_device, driver_name):
        if not drm_device or not os.path.exists(drm_device):
            raise FileNotFoundError(f"Intel DRM device not found: {drm_device}")

        os.environ["GST_VAAPI_DRM_DEVICE"] = drm_device
        if driver_name:
            os.environ["LIBVA_DRIVER_NAME"] = driver_name
        else:
            os.environ.pop("LIBVA_DRIVER_NAME", None)

    def _build_pipeline(self, decoder_name, use_hw_decode):
        protocols = getattr(self.config, "RTSP_PROTOCOLS", "tcp")
        latency = int(getattr(self.config, "RTSP_LATENCY_MS", 200))
        drop_on_latency = "true" if use_hw_decode else "false"

        if use_hw_decode:
            if decoder_name == "vah264dec":
                decode_stage = (
                    "vah264dec ! "
                    "vapostproc ! video/x-raw,format=BGRx ! "
                    "videoconvert n-threads=1 ! video/x-raw,format=BGR"
                )
            else:
                decode_stage = (
                    f"{decoder_name} low-latency=true ! "
                    "video/x-raw,format=BGRx ! "
                    "videoconvert n-threads=1 ! video/x-raw,format=BGR"
                )
        elif decoder_name == "decodebin":
            decode_stage = "decodebin ! videoconvert n-threads=1 ! video/x-raw,format=BGR"
        else:
            decode_stage = f"{decoder_name} max-threads=1 ! videoconvert n-threads=1 ! video/x-raw,format=BGR"

        return (
            f'rtspsrc location="{self.rtsp_url}" protocols={protocols} latency={latency} '
            f'drop-on-latency={drop_on_latency} name=src '
            "src. ! queue max-size-buffers=8 max-size-bytes=0 max-size-time=0 "
            "! application/x-rtp,media=video,encoding-name=H264 "
            f"! rtpjitterbuffer latency={latency} "
            "! rtph264depay ! h264parse config-interval=-1 disable-passthrough=true "
            f"! {decode_stage} "
            "! queue max-size-buffers=4 max-size-bytes=0 max-size-time=0 "
            "! appsink drop=true sync=false max-buffers=1 enable-last-sample=false wait-on-eos=false"
        )

    def _open_ffmpeg_capture(self):
        capture_options = []
        protocols = getattr(self.config, "RTSP_PROTOCOLS", "").strip().lower()
        if protocols in {"tcp", "udp"}:
            capture_options.append(f"rtsp_transport;{protocols}")

        extra_options = getattr(self.config, "FFMPEG_CAPTURE_OPTIONS", "").strip()
        if extra_options:
            capture_options.append(extra_options)

        if capture_options:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "|".join(capture_options)

        logging.info("Trying CPU RTSP capture with OpenCV FFmpeg backend")
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        if cap.isOpened() and cap.grab():
            self.cap = cap
            self.is_connected = True
            self.active_decoder = "opencv_ffmpeg"
            self.active_decode_mode = "cpu"
            logging.info("RTSP opened successfully with CPU decoder 'opencv_ffmpeg'")
            return True

        if cap:
            cap.release()
        logging.error("CPU decoder 'opencv_ffmpeg' failed to produce frames")
        return False

    def _try_decoder_group(self, decoder_candidates, use_hw_decode, drm_device=None, driver_name=None):
        if use_hw_decode and drm_device:
            try:
                self._configure_intel_decode(drm_device, driver_name)
            except FileNotFoundError as exc:
                logging.error(str(exc))
                return False
        elif not use_hw_decode:
            self._reset_intel_decode_env()

        for decoder_name in decoder_candidates:
            if not self._has_gstreamer_element(decoder_name):
                logging.warning("Skipping unavailable GStreamer decoder '%s'", decoder_name)
                continue

            pipeline = self._build_pipeline(decoder_name, use_hw_decode=use_hw_decode)
            if use_hw_decode:
                logging.info(
                    "Trying Intel RTSP pipeline with decoder '%s' using driver '%s' on %s",
                    decoder_name,
                    driver_name,
                    drm_device,
                )
            else:
                logging.info("Trying CPU RTSP pipeline with decoder '%s'", decoder_name)

            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened() and cap.grab():
                self.cap = cap
                self.is_connected = True
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.active_decoder = decoder_name
                self.active_decode_mode = "hardware" if use_hw_decode else "cpu"
                self.active_drm_device = drm_device if use_hw_decode else None
                self.active_libva_driver = driver_name if use_hw_decode else None
                logging.info(
                    "RTSP opened successfully with %s decoder '%s'%s",
                    self.active_decode_mode,
                    decoder_name,
                    f" (driver={driver_name}, device={drm_device})" if use_hw_decode else "",
                )
                return True

            if cap:
                cap.release()
            logging.error(
                "%s decoder '%s' failed to produce frames%s",
                "Intel hardware" if use_hw_decode else "CPU",
                decoder_name,
                f" (driver={driver_name}, device={drm_device})" if use_hw_decode else "",
            )

        return False

    def _open_capture(self):
        self.active_decoder = None
        self.active_decode_mode = None
        self.active_drm_device = None
        self.active_libva_driver = None
        self._reset_intel_decode_env()

        prefer_hw_decode = getattr(self.config, "PREFER_INTEL_HW_DECODE", True)
        allow_cpu_fallback = getattr(self.config, "ALLOW_CPU_FALLBACK", True)

        if prefer_hw_decode:
            drm_devices = self._discover_intel_drm_devices()
            libva_driver_candidates = getattr(self.config, "LIBVA_DRIVER_CANDIDATES", ("iHD", "i965"))
            if drm_devices:
                for drm_device in drm_devices:
                    for driver_name in libva_driver_candidates:
                        if self._try_decoder_group(
                            getattr(self.config, "GSTREAMER_HW_DECODERS", ("vaapih264dec",)),
                            use_hw_decode=True,
                            drm_device=drm_device,
                            driver_name=driver_name,
                        ):
                            return True
            else:
                logging.warning("No Intel render device detected for hardware decode")

        if allow_cpu_fallback:
            self._reset_intel_decode_env()
            if getattr(self.config, "PREFER_FFMPEG_CPU_FALLBACK", True):
                if self._open_ffmpeg_capture():
                    return True
            if self._try_decoder_group(
                getattr(self.config, "GSTREAMER_SW_DECODERS", ("avdec_h264",)),
                use_hw_decode=False,
            ):
                return True

        if prefer_hw_decode and not allow_cpu_fallback:
            logging.error("No Intel hardware decoder pipeline could be opened. CPU fallback is disabled.")
        else:
            logging.error("No RTSP decoder pipeline could be opened.")
        self.is_connected = False
        return False

    def connect(self):
        return self._open_capture()

    def reconnect(self):
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        return self._open_capture()

    def check_health(self):
        return bool(self.cap and self.cap.isOpened() and self.is_connected)

    def read(self):
        if not self.cap:
            return False, None
        return self.cap.read()

    def close(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self._reset_intel_decode_env()
        self.is_connected = False


class LaneDetector:
    def __init__(self, config):
        self.config = config
        self.model = YOLO(config.MODEL_PATH)
        self.model_task = getattr(self.model, "task", None) or getattr(self.model.model, "task", None) or "detect"
        polygon_classes = [
            class_name
            for class_name, annotation_type in zip(config.CLASS_NAMES, config.CLASS_ANNOTATION_TYPES)
            if annotation_type == "polygon"
        ]
        if self.model_task == "detect" and polygon_classes:
            logging.warning(
                "Model task is 'detect'; polygon classes %s will render as bbox until a segmentation model is loaded",
                ", ".join(polygon_classes),
            )

    def _extract_polygon(self, result, index, roi_offset_x, roi_offset_y):
        masks = getattr(result, "masks", None)
        if masks is None or masks.xy is None or index >= len(masks.xy):
            return None

        polygon = masks.xy[index]
        if polygon is None or len(polygon) < 3:
            return None

        points = []
        for x_value, y_value in polygon:
            points.append([
                int(round(float(x_value) + roi_offset_x)),
                int(round(float(y_value) + roi_offset_y)),
            ])
        return points if len(points) >= 3 else None

    def detect(self, image, roi_coords=None):
        roi_offset_x = 0
        roi_offset_y = 0
        inference_image = image

        if roi_coords is not None:
            x1, y1, x2, y2 = roi_coords
            if x2 <= x1 or y2 <= y1:
                return []
            inference_image = image[y1:y2, x1:x2]
            roi_offset_x = x1
            roi_offset_y = y1

        results = self.model.predict(
            source=inference_image,
            imgsz=self.config.MODEL_IMAGE_SIZE,
            conf=self.config.CONFIDENCE_THRESHOLD,
            iou=self.config.NMS_THRESHOLD,
            device=self.config.MODEL_DEVICE,
            max_det=self.config.MAX_DETECTIONS,
            verbose=False,
        )

        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        detections = []
        boxes = boxes.cpu()

        for index, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            class_id = int(box.cls[0].item()) if box.cls is not None else 0
            annotation_type = (
                self.config.CLASS_ANNOTATION_TYPES[class_id]
                if 0 <= class_id < len(self.config.CLASS_ANNOTATION_TYPES)
                else "bbox"
            )
            polygon = None
            if annotation_type == "polygon" and self.model_task == "segment":
                polygon = self._extract_polygon(result=results[0], index=index, roi_offset_x=roi_offset_x, roi_offset_y=roi_offset_y)

            detections.append(
                {
                    "bbox": [x1 + roi_offset_x, y1 + roi_offset_y, x2 + roi_offset_x, y2 + roi_offset_y],
                    "confidence": float(box.conf[0].item()) if box.conf is not None else 0.0,
                    "class_id": class_id,
                    "annotation_type": annotation_type,
                    "polygon": polygon,
                    "render_as": "polygon" if polygon is not None and annotation_type == "polygon" else "bbox",
                }
            )

        return detections


class AsyncLaneDetector:
    def __init__(self, config):
        self.config = config
        self.frame_buffer = FrameBuffer(maxsize=config.DETECTOR_INPUT_QUEUE_SIZE)
        self.result_buffer = FrameBuffer(maxsize=config.DETECTOR_RESULT_QUEUE_SIZE)
        self.processing = False
        self.detection_thread = None
        self.detector = LaneDetector(config)

    def start_processing(self):
        if self.processing:
            return
        self.processing = True
        self.detection_thread = threading.Thread(target=self._process_frames, daemon=True)
        self.detection_thread.start()

    def _process_frames(self):
        while self.processing:
            item = self.frame_buffer.get(timeout=self.config.DETECTOR_POLL_TIMEOUT_S)
            if item is None:
                continue

            frame, roi_coords = item
            start_time = time.time()

            try:
                detections = self.detector.detect(frame, roi_coords)
                process_time = time.time() - start_time
                self.result_buffer.put((frame.copy(), detections, roi_coords, process_time, time.time()))
            except Exception:
                logging.exception("Lane detection failed")

    def add_frame(self, frame, roi_coords):
        self.frame_buffer.put((frame, roi_coords))

    def get_result(self):
        return self.result_buffer.get()

    def stop(self):
        self.processing = False
        if self.detection_thread:
            self.detection_thread.join()
