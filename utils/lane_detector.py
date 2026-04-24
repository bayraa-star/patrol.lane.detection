import logging
import os
from pathlib import Path
import subprocess
import threading
import time
from queue import Empty, Queue

import cv2
import numpy as np


COCO_CLASS_NAMES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
)


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


def _get_detection_center(detection):
    x1, y1, x2, y2 = detection["bbox"]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _bbox_iou(first_bbox, second_bbox):
    ax1, ay1, ax2, ay2 = first_bbox
    bx1, by1, bx2, by2 = second_bbox
    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)
    intersection_width = max(0, intersection_x2 - intersection_x1)
    intersection_height = max(0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height

    first_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    second_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area = first_area + second_area - intersection_area
    if union_area <= 0:
        return 0.0
    return intersection_area / union_area


def _get_class_name(class_names, class_id):
    return class_names[class_id] if 0 <= class_id < len(class_names) else f"class_{class_id}"


def _get_best_detection_by_class(detections, class_names, target_class_name):
    target_class_ids = {
        class_id
        for class_id, class_name in enumerate(class_names)
        if class_name == target_class_name
    }
    if not target_class_ids:
        return None

    matching_detections = [
        detection
        for detection in detections
        if detection.get("class_id") in target_class_ids
    ]
    if not matching_detections:
        return None

    return max(matching_detections, key=lambda detection: detection.get("confidence", 0.0))


def _get_lane_line_points(detection_bbox, patrol_center_x):
    x1, y1, x2, y2 = detection_bbox
    detection_center_x = (x1 + x2) / 2.0

    if detection_center_x < patrol_center_x:
        return (x1, y2), (x2, y1)
    return (x2, y2), (x1, y1)


def _extend_line_to_frame_coverage(image_shape, start_point, end_point, coverage_ratio):
    height, width = image_shape[:2]
    dx = float(end_point[0] - start_point[0])
    dy = float(end_point[1] - start_point[1])
    current_length = float(np.hypot(dx, dy))
    if current_length <= 1e-6:
        return start_point, end_point

    target_length = coverage_ratio * float(np.hypot(width, height))
    if target_length <= current_length:
        return start_point, end_point

    scale = target_length / current_length
    extended_end_point = (
        int(round(start_point[0] + dx * scale)),
        int(round(start_point[1] + dy * scale)),
    )

    clipped, clipped_start, clipped_end = cv2.clipLine(
        (0, 0, width, height),
        start_point,
        extended_end_point,
    )
    if not clipped:
        return start_point, end_point

    if 0 <= start_point[0] < width and 0 <= start_point[1] < height:
        clipped_start = start_point

    return clipped_start, clipped_end


def _get_nearest_lane_detections(detections, class_names, patrol_center_x):
    nearest_by_side = {"left": None, "right": None}
    nearest_distance_by_side = {"left": float("inf"), "right": float("inf")}

    for detection in detections:
        class_id = detection.get("class_id", -1)
        class_name = _get_class_name(class_names, class_id)
        if class_name == "patrol":
            continue

        detection_center_x, _ = _get_detection_center(detection)
        if detection_center_x < patrol_center_x:
            side = "left"
        elif detection_center_x > patrol_center_x:
            side = "right"
        else:
            continue

        distance = abs(detection_center_x - patrol_center_x)
        current_detection = nearest_by_side[side]
        current_distance = nearest_distance_by_side[side]

        if current_detection is None or distance < current_distance:
            nearest_by_side[side] = detection
            nearest_distance_by_side[side] = distance
            continue

        if distance == current_distance and detection.get("confidence", 0.0) > current_detection.get("confidence", 0.0):
            nearest_by_side[side] = detection
            nearest_distance_by_side[side] = distance

    return nearest_by_side


def _get_sorted_right_side_detections(detections, class_names, patrol_center_x):
    right_side_detections = []

    for detection in detections:
        class_id = detection.get("class_id", -1)
        class_name = _get_class_name(class_names, class_id)
        if class_name == "patrol":
            continue

        detection_center_x, _ = _get_detection_center(detection)
        if detection_center_x <= patrol_center_x:
            continue

        right_side_detections.append(
            (
                abs(detection_center_x - patrol_center_x),
                -float(detection.get("confidence", 0.0)),
                detection,
            )
        )

    right_side_detections.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in right_side_detections]


def get_patrol_lane_info(detections, class_names):
    patrol_detection = _get_best_detection_by_class(detections, class_names, "patrol")
    if patrol_detection is None:
        return None

    patrol_center_x, _ = _get_detection_center(patrol_detection)
    right_side_detections = _get_sorted_right_side_detections(detections, class_names, patrol_center_x)
    if not right_side_detections:
        return {
            "lane_id": None,
            "label": "Unknown",
            "nearest_right_class": None,
            "second_right_class": None,
        }

    nearest_right = right_side_detections[0]
    nearest_right_name = _get_class_name(class_names, nearest_right.get("class_id", -1))

    if nearest_right_name == "curve":
        return {
            "lane_id": 1,
            "label": "Lane 1",
            "nearest_right_class": nearest_right_name,
            "second_right_class": None,
        }

    if nearest_right_name not in {"solid", "dashed"} or len(right_side_detections) < 2:
        return {
            "lane_id": None,
            "label": "Unknown",
            "nearest_right_class": nearest_right_name,
            "second_right_class": None,
        }

    second_nearest_right = right_side_detections[1]
    second_nearest_right_name = _get_class_name(class_names, second_nearest_right.get("class_id", -1))
    if second_nearest_right_name != "curve":
        return {
            "lane_id": None,
            "label": "Unknown",
            "nearest_right_class": nearest_right_name,
            "second_right_class": second_nearest_right_name,
        }

    return {
        "lane_id": 2,
        "label": "Lane 2",
        "nearest_right_class": nearest_right_name,
        "second_right_class": second_nearest_right_name,
    }


def draw_patrol_lane_status(image, lane_info, line_thickness=2, text_scale=0.8, text_thickness=2):
    if not lane_info:
        return

    label = f"Patrol Moving: {lane_info['label']}"
    (label_width, label_height), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        text_scale,
        text_thickness,
    )
    x1 = 16
    y1 = 16
    x2 = x1 + label_width + 16
    y2 = y1 + label_height + baseline + 16

    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 220, 255), thickness=-1)
    cv2.addWeighted(overlay, 0.25, image, 0.75, 0.0, image)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 255), max(2, line_thickness), cv2.LINE_AA)
    cv2.putText(
        image,
        label,
        (x1 + 8, y2 - baseline - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        text_scale,
        (0, 0, 0),
        text_thickness,
        cv2.LINE_AA,
    )


def _get_lane_guide_segment(image_shape, detection_bbox, patrol_center_x, coverage_ratio):
    start_point, end_point = _get_lane_line_points(detection_bbox, patrol_center_x)
    return _extend_line_to_frame_coverage(
        image_shape,
        start_point,
        end_point,
        coverage_ratio,
    )


def _draw_lane_fill(image, left_segment, right_segment, fill_color=(0, 255, 0), fill_alpha=0.20):
    if left_segment is None or right_segment is None:
        return

    overlay = image.copy()
    polygon = np.array(
        [
            left_segment[0],
            left_segment[1],
            right_segment[1],
            right_segment[0],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(overlay, [polygon], fill_color)
    cv2.addWeighted(overlay, fill_alpha, image, 1.0 - fill_alpha, 0.0, image)


def draw_lane_guides(image, detections, class_names, class_colors, line_thickness, coverage_ratio, fill_alpha):
    patrol_detection = _get_best_detection_by_class(detections, class_names, "patrol")
    if patrol_detection is None:
        return

    patrol_center_x, _ = _get_detection_center(patrol_detection)
    nearest_detections = _get_nearest_lane_detections(detections, class_names, patrol_center_x)
    guide_segments = {}

    for side, lane_detection in nearest_detections.items():
        if lane_detection is None:
            continue

        guide_segments[side] = _get_lane_guide_segment(
            image.shape,
            lane_detection["bbox"],
            patrol_center_x,
            coverage_ratio,
        )

    _draw_lane_fill(
        image,
        guide_segments.get("left"),
        guide_segments.get("right"),
        fill_alpha=fill_alpha,
    )

    for side, lane_detection in nearest_detections.items():
        if lane_detection is None:
            continue

        start_point, end_point = guide_segments[side]
        class_id = lane_detection["class_id"]
        color = class_colors[class_id % len(class_colors)] if class_colors else (0, 255, 0)
        cv2.line(
            image,
            start_point,
            end_point,
            color,
            max(2, line_thickness + 1),
            cv2.LINE_AA,
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
    lane_guide_coverage_ratio=0.70,
    lane_guide_fill_alpha=0.20,
):
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        class_id = detection["class_id"]
        confidence = detection["confidence"]
        polygon = detection.get("polygon")
        render_as = detection.get("render_as", "bbox")

        color = class_colors[class_id % len(class_colors)] if class_colors else (0, 255, 0)
        class_name = _get_class_name(class_names, class_id)
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

    draw_lane_guides(
        image,
        detections,
        class_names,
        class_colors,
        line_thickness,
        lane_guide_coverage_ratio,
        lane_guide_fill_alpha,
    )
    draw_patrol_lane_status(
        image,
        get_patrol_lane_info(detections, class_names),
        line_thickness=line_thickness,
        text_scale=max(0.7, text_scale + 0.1),
        text_thickness=text_thickness,
    )


def draw_object_detections(
    image,
    detections,
    class_names,
    class_colors,
    line_thickness=2,
    text_scale=0.6,
    text_thickness=2,
):
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        class_id = detection["class_id"]
        confidence = detection["confidence"]

        color = class_colors[class_id % len(class_colors)] if class_colors else (0, 255, 255)
        class_name = _get_class_name(class_names, class_id)
        track_id = detection.get("track_id")
        if track_id is None:
            label = f"{class_name} {confidence:.2f}"
        else:
            label = f"{class_name} #{track_id} {confidence:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness, cv2.LINE_AA)
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


def _normalize_openvino_device(device_name):
    if not device_name:
        return "CPU"
    normalized = str(device_name).strip()
    return "CPU" if normalized.lower() == "cpu" else normalized


def _openvino_compile_config(config):
    compile_config = {}
    performance_hint = getattr(config, "PERFORMANCE_HINT", "").strip().upper()
    if performance_hint:
        compile_config["PERFORMANCE_HINT"] = performance_hint

    num_streams = str(getattr(config, "OPENVINO_NUM_STREAMS", "")).strip()
    if num_streams:
        compile_config["NUM_STREAMS"] = num_streams

    inference_threads = int(getattr(config, "OPENVINO_INFERENCE_NUM_THREADS", 0))
    if inference_threads > 0:
        compile_config["INFERENCE_NUM_THREADS"] = inference_threads

    return compile_config


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
                    "vapostproc ! video/x-raw,format=NV12 ! "
                    "videoconvert n-threads=1 ! video/x-raw,format=BGR"
                )
            else:
                decode_stage = (
                    f"{decoder_name} low-latency=true ! "
                    "vaapipostproc ! video/x-raw,format=NV12 ! "
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


class UltralyticsLaneDetector:
    def __init__(self, config):
        from ultralytics import YOLO

        self.config = config
        self.model = YOLO(config.MODEL_PATH)
        self.model_task = getattr(self.model, "task", None) or getattr(self.model.model, "task", None) or "detect"
        raw_model_names = getattr(self.model, "names", None) or getattr(self.model.model, "names", None) or {}
        if isinstance(raw_model_names, dict):
            self.model_class_names = [str(raw_model_names[index]) for index in sorted(raw_model_names)]
        elif isinstance(raw_model_names, list):
            self.model_class_names = [str(name) for name in raw_model_names]
        else:
            self.model_class_names = []

        if self.model_class_names:
            if self.model_class_names == config.CLASS_NAMES:
                logging.info("Model class names match configs.yaml: %s", ", ".join(self.model_class_names))
            else:
                logging.warning(
                    "Model/config class name mismatch. Model=%s Config=%s",
                    ", ".join(self.model_class_names),
                    ", ".join(config.CLASS_NAMES),
                )

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


class OpenVINOLaneDetector:
    def __init__(self, config):
        import openvino as ov
        import torch
        from ultralytics.utils.nms import non_max_suppression
        from ultralytics.utils.ops import scale_boxes

        self.config = config
        self.ov = ov
        self.torch = torch
        self.non_max_suppression = non_max_suppression
        self.scale_boxes = scale_boxes
        self.model_task = "detect"
        self.model_class_names = list(config.CLASS_NAMES)
        torch_threads = int(getattr(config, "TORCH_NUM_THREADS", 0))
        if torch_threads > 0:
            self.torch.set_num_threads(torch_threads)

        self.core = ov.Core()
        self.device_name = self._normalize_device(config.MODEL_DEVICE)
        self.model = self.core.read_model(config.MODEL_PATH)
        self.compile_config = _openvino_compile_config(config)
        self.compiled_model = self.core.compile_model(self.model, self.device_name, self.compile_config)
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
        self.input_shape = tuple(int(dimension) for dimension in self.input_layer.shape)
        if len(self.input_shape) != 4:
            raise ValueError(f"Expected OpenVINO model input shape [N,C,H,W], got {self.input_shape}")
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

        logging.info(
            "Loaded OpenVINO lane detector from %s on device %s with input shape %s and config %s",
            config.MODEL_PATH,
            self.device_name,
            self.input_shape,
            self.compile_config,
        )

    @staticmethod
    def _normalize_device(device_name):
        if not device_name:
            return "CPU"
        normalized = str(device_name).strip()
        return "CPU" if normalized.lower() == "cpu" else normalized

    def _letterbox(self, image):
        image_height, image_width = image.shape[:2]
        gain = min(self.input_width / image_width, self.input_height / image_height)
        resized_width = int(round(image_width * gain))
        resized_height = int(round(image_height * gain))

        if (resized_width, resized_height) != (image_width, image_height):
            resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        else:
            resized = image

        pad_width = self.input_width - resized_width
        pad_height = self.input_height - resized_height
        pad_left = int(round(pad_width / 2 - 0.1))
        pad_right = int(round(pad_width / 2 + 0.1))
        pad_top = int(round(pad_height / 2 - 0.1))
        pad_bottom = int(round(pad_height / 2 + 0.1))

        bordered = cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        ratio_pad = ((gain, gain), (pad_left, pad_top))
        return bordered, ratio_pad

    def _preprocess(self, image):
        letterboxed, ratio_pad = self._letterbox(image)
        input_tensor = letterboxed[:, :, ::-1].transpose(2, 0, 1)
        input_tensor = np.ascontiguousarray(input_tensor, dtype=np.float32) / 255.0
        return input_tensor[None], letterboxed.shape[:2], ratio_pad

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

        input_tensor, processed_shape, ratio_pad = self._preprocess(inference_image)
        raw_prediction = self.compiled_model(input_tensor)[self.output_layer]
        prediction = self.torch.from_numpy(raw_prediction)
        batches = self.non_max_suppression(
            prediction,
            conf_thres=self.config.CONFIDENCE_THRESHOLD,
            iou_thres=self.config.NMS_THRESHOLD,
            max_det=self.config.MAX_DETECTIONS,
            nc=len(self.config.CLASS_NAMES),
        )
        if not batches or batches[0].shape[0] == 0:
            return []

        boxes = batches[0].cpu()
        boxes[:, :4] = self.scale_boxes(
            processed_shape,
            boxes[:, :4],
            inference_image.shape[:2],
            ratio_pad=ratio_pad,
        )

        image_height, image_width = inference_image.shape[:2]
        detections = []
        for box in boxes:
            x1, y1, x2, y2, confidence, class_id = box[:6].tolist()
            class_id = int(class_id)
            annotation_type = (
                self.config.CLASS_ANNOTATION_TYPES[class_id]
                if 0 <= class_id < len(self.config.CLASS_ANNOTATION_TYPES)
                else "bbox"
            )
            x1 = int(round(max(0, min(x1, image_width - 1))))
            y1 = int(round(max(0, min(y1, image_height - 1))))
            x2 = int(round(max(0, min(x2, image_width - 1))))
            y2 = int(round(max(0, min(y2, image_height - 1))))
            if x2 <= x1 or y2 <= y1:
                continue

            detections.append(
                {
                    "bbox": [x1 + roi_offset_x, y1 + roi_offset_y, x2 + roi_offset_x, y2 + roi_offset_y],
                    "confidence": float(confidence),
                    "class_id": class_id,
                    "annotation_type": annotation_type,
                    "polygon": None,
                    "render_as": "bbox",
                }
            )

        return detections


class OpenVINOVehicleDetector:
    def __init__(self, config):
        import openvino as ov
        import torch
        from ultralytics.utils.nms import non_max_suppression
        from ultralytics.utils.ops import scale_boxes

        self.config = config
        self.ov = ov
        self.torch = torch
        self.non_max_suppression = non_max_suppression
        self.scale_boxes = scale_boxes
        self.model_task = "detect"
        self.model_class_names = list(config.VEHICLE_CLASS_NAMES)
        torch_threads = int(getattr(config, "TORCH_NUM_THREADS", 0))
        if torch_threads > 0:
            self.torch.set_num_threads(torch_threads)
        self.available_class_names = list(COCO_CLASS_NAMES)
        self.target_class_ids = {
            class_id
            for class_id, class_name in enumerate(self.available_class_names)
            if class_name in set(config.VEHICLE_CLASS_NAMES)
        }
        self.display_class_id_by_name = {
            class_name: index
            for index, class_name in enumerate(config.VEHICLE_CLASS_NAMES)
        }

        self.core = ov.Core()
        self.device_name = self._normalize_device(config.VEHICLE_MODEL_DEVICE)
        self.model = self.core.read_model(config.VEHICLE_MODEL_PATH)
        self.compile_config = _openvino_compile_config(config)
        self.compiled_model = self.core.compile_model(self.model, self.device_name, self.compile_config)
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
        self.input_shape = tuple(int(dimension) for dimension in self.input_layer.shape)
        if len(self.input_shape) != 4:
            raise ValueError(f"Expected OpenVINO model input shape [N,C,H,W], got {self.input_shape}")
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

        logging.info(
            "Loaded OpenVINO vehicle detector from %s on device %s with input shape %s and config %s",
            config.VEHICLE_MODEL_PATH,
            self.device_name,
            self.input_shape,
            self.compile_config,
        )

    @staticmethod
    def _normalize_device(device_name):
        return _normalize_openvino_device(device_name)

    def _letterbox(self, image):
        image_height, image_width = image.shape[:2]
        gain = min(self.input_width / image_width, self.input_height / image_height)
        resized_width = int(round(image_width * gain))
        resized_height = int(round(image_height * gain))

        if (resized_width, resized_height) != (image_width, image_height):
            resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        else:
            resized = image

        pad_width = self.input_width - resized_width
        pad_height = self.input_height - resized_height
        pad_left = int(round(pad_width / 2 - 0.1))
        pad_right = int(round(pad_width / 2 + 0.1))
        pad_top = int(round(pad_height / 2 - 0.1))
        pad_bottom = int(round(pad_height / 2 + 0.1))

        bordered = cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        ratio_pad = ((gain, gain), (pad_left, pad_top))
        return bordered, ratio_pad

    def _preprocess(self, image):
        letterboxed, ratio_pad = self._letterbox(image)
        input_tensor = letterboxed[:, :, ::-1].transpose(2, 0, 1)
        input_tensor = np.ascontiguousarray(input_tensor, dtype=np.float32) / 255.0
        return input_tensor[None], letterboxed.shape[:2], ratio_pad

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

        input_tensor, processed_shape, ratio_pad = self._preprocess(inference_image)
        raw_prediction = self.compiled_model(input_tensor)[self.output_layer]
        prediction = self.torch.from_numpy(raw_prediction)
        batches = self.non_max_suppression(
            prediction,
            conf_thres=self.config.VEHICLE_CONFIDENCE_THRESHOLD,
            iou_thres=self.config.VEHICLE_NMS_THRESHOLD,
            max_det=self.config.VEHICLE_MAX_DETECTIONS,
            nc=len(self.available_class_names),
        )
        if not batches or batches[0].shape[0] == 0:
            return []

        boxes = batches[0].cpu()
        boxes[:, :4] = self.scale_boxes(
            processed_shape,
            boxes[:, :4],
            inference_image.shape[:2],
            ratio_pad=ratio_pad,
        )

        image_height, image_width = inference_image.shape[:2]
        detections = []
        for box in boxes:
            x1, y1, x2, y2, confidence, class_id = box[:6].tolist()
            source_class_id = int(class_id)
            if source_class_id not in self.target_class_ids:
                continue

            source_class_name = self.available_class_names[source_class_id]
            display_class_id = self.display_class_id_by_name[source_class_name]
            x1 = int(round(max(0, min(x1, image_width - 1))))
            y1 = int(round(max(0, min(y1, image_height - 1))))
            x2 = int(round(max(0, min(x2, image_width - 1))))
            y2 = int(round(max(0, min(y2, image_height - 1))))
            if x2 <= x1 or y2 <= y1:
                continue

            detections.append(
                {
                    "bbox": [x1 + roi_offset_x, y1 + roi_offset_y, x2 + roi_offset_x, y2 + roi_offset_y],
                    "confidence": float(confidence),
                    "class_id": display_class_id,
                }
            )

        return detections


class UltralyticsVehicleDetector:
    def __init__(self, config):
        from ultralytics import YOLO

        self.config = config
        self.model = YOLO(config.VEHICLE_MODEL_PATH)
        self.model_task = getattr(self.model, "task", None) or getattr(self.model.model, "task", None) or "detect"
        raw_model_names = getattr(self.model, "names", None) or getattr(self.model.model, "names", None) or {}
        if isinstance(raw_model_names, dict):
            self.available_class_names = [str(raw_model_names[index]) for index in sorted(raw_model_names)]
        elif isinstance(raw_model_names, list):
            self.available_class_names = [str(name) for name in raw_model_names]
        else:
            self.available_class_names = []

        self.target_class_names = set(config.VEHICLE_CLASS_NAMES)
        self.display_class_id_by_name = {
            class_name: index
            for index, class_name in enumerate(config.VEHICLE_CLASS_NAMES)
        }
        self.model_class_names = list(config.VEHICLE_CLASS_NAMES)

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
            imgsz=self.config.VEHICLE_MODEL_IMAGE_SIZE,
            conf=self.config.VEHICLE_CONFIDENCE_THRESHOLD,
            iou=self.config.VEHICLE_NMS_THRESHOLD,
            device=self.config.VEHICLE_MODEL_DEVICE,
            max_det=self.config.VEHICLE_MAX_DETECTIONS,
            verbose=False,
        )
        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        detections = []
        boxes = boxes.cpu()

        for box in boxes:
            source_class_id = int(box.cls[0].item()) if box.cls is not None else -1
            if not (0 <= source_class_id < len(self.available_class_names)):
                continue

            source_class_name = self.available_class_names[source_class_id]
            if source_class_name not in self.target_class_names:
                continue

            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            detections.append(
                {
                    "bbox": [x1 + roi_offset_x, y1 + roi_offset_y, x2 + roi_offset_x, y2 + roi_offset_y],
                    "confidence": float(box.conf[0].item()) if box.conf is not None else 0.0,
                    "class_id": self.display_class_id_by_name[source_class_name],
                }
            )

        return detections


class VehicleDetector:
    def __init__(self, config):
        self.config = config
        self.backend_name = self._resolve_backend(config.VEHICLE_MODEL_BACKEND, config.VEHICLE_MODEL_PATH)
        if self.backend_name == "openvino":
            self.backend = OpenVINOVehicleDetector(config)
        else:
            self.backend = UltralyticsVehicleDetector(config)

        self.model_task = getattr(self.backend, "model_task", "detect")
        self.model_class_names = getattr(self.backend, "model_class_names", [])
        logging.info("Vehicle detector backend: %s", self.backend_name)

    @staticmethod
    def _resolve_backend(configured_backend, model_path):
        if configured_backend == "auto":
            return "openvino" if Path(model_path).suffix.lower() == ".xml" else "ultralytics"
        return configured_backend

    def detect(self, image, roi_coords=None):
        return self.backend.detect(image, roi_coords)


class ByteTrackVehicleTracker:
    def __init__(self, config):
        self.high_threshold = float(getattr(config, "BYTE_TRACK_HIGH_THRESHOLD", 0.45))
        self.low_threshold = float(getattr(config, "BYTE_TRACK_LOW_THRESHOLD", 0.10))
        self.match_threshold = float(getattr(config, "BYTE_TRACK_MATCH_THRESHOLD", 0.30))
        self.max_lost_frames = int(getattr(config, "BYTE_TRACK_MAX_LOST_FRAMES", 30))
        self.next_track_id = 1
        self.tracks = []

    def update(self, detections):
        high_detections = [
            dict(detection)
            for detection in detections
            if detection.get("confidence", 0.0) >= self.high_threshold
        ]
        low_detections = [
            dict(detection)
            for detection in detections
            if self.low_threshold <= detection.get("confidence", 0.0) < self.high_threshold
        ]

        for track in self.tracks:
            track["lost_frames"] += 1
            track["updated"] = False

        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_high = list(range(len(high_detections)))
        _, unmatched_tracks, unmatched_high = self._match(
            unmatched_tracks,
            high_detections,
            unmatched_high,
        )

        unmatched_low = list(range(len(low_detections)))
        _, unmatched_tracks, _ = self._match(unmatched_tracks, low_detections, unmatched_low)

        for detection_index in unmatched_high:
            self._start_track(high_detections[detection_index])

        self.tracks = [
            track
            for track in self.tracks
            if track["lost_frames"] <= self.max_lost_frames
        ]

        return [
            self._tracked_detection(track)
            for track in self.tracks
            if track["updated"] or track["lost_frames"] <= 1
        ]

    def _match(self, track_indices, detections, detection_indices):
        pairs = []
        for track_index in track_indices:
            track = self.tracks[track_index]
            for detection_index in detection_indices:
                detection = detections[detection_index]
                if detection.get("class_id") != track["class_id"]:
                    continue
                iou = _bbox_iou(track["bbox"], detection["bbox"])
                if iou >= self.match_threshold:
                    pairs.append((iou, track_index, detection_index))

        pairs.sort(reverse=True, key=lambda item: item[0])
        matched_track_indices = set()
        matched_detection_indices = set()
        for _, track_index, detection_index in pairs:
            if track_index in matched_track_indices or detection_index in matched_detection_indices:
                continue
            self._update_track(self.tracks[track_index], detections[detection_index])
            matched_track_indices.add(track_index)
            matched_detection_indices.add(detection_index)

        unmatched_tracks = [
            track_index
            for track_index in track_indices
            if track_index not in matched_track_indices
        ]
        unmatched_detections = [
            detection_index
            for detection_index in detection_indices
            if detection_index not in matched_detection_indices
        ]
        return matched_track_indices, unmatched_tracks, unmatched_detections

    def _start_track(self, detection):
        track = {
            "track_id": self.next_track_id,
            "bbox": detection["bbox"],
            "confidence": detection.get("confidence", 0.0),
            "class_id": detection.get("class_id", -1),
            "lost_frames": 0,
            "updated": True,
        }
        self.next_track_id += 1
        self.tracks.append(track)

    @staticmethod
    def _update_track(track, detection):
        track["bbox"] = detection["bbox"]
        track["confidence"] = detection.get("confidence", track["confidence"])
        track["class_id"] = detection.get("class_id", track["class_id"])
        track["lost_frames"] = 0
        track["updated"] = True

    @staticmethod
    def _tracked_detection(track):
        return {
            "bbox": list(track["bbox"]),
            "confidence": float(track["confidence"]),
            "class_id": int(track["class_id"]),
            "track_id": int(track["track_id"]),
            "lost_frames": int(track["lost_frames"]),
        }


class LaneDetector:
    def __init__(self, config):
        self.config = config
        self.backend_name = self._resolve_backend(config.MODEL_BACKEND, config.MODEL_PATH)
        if self.backend_name == "openvino":
            self.backend = OpenVINOLaneDetector(config)
        else:
            self.backend = UltralyticsLaneDetector(config)

        self.model_task = getattr(self.backend, "model_task", "detect")
        self.model_class_names = getattr(self.backend, "model_class_names", [])
        logging.info("Lane detector backend: %s", self.backend_name)

    @staticmethod
    def _resolve_backend(configured_backend, model_path):
        if configured_backend == "auto":
            return "openvino" if Path(model_path).suffix.lower() == ".xml" else "ultralytics"
        return configured_backend

    def detect(self, image, roi_coords=None):
        return self.backend.detect(image, roi_coords)


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


class AsyncVehicleDetector:
    def __init__(self, config):
        self.config = config
        self.frame_buffer = FrameBuffer(maxsize=config.DETECTOR_INPUT_QUEUE_SIZE)
        self.result_buffer = FrameBuffer(maxsize=config.DETECTOR_RESULT_QUEUE_SIZE)
        self.processing = False
        self.detection_thread = None
        self.detector = VehicleDetector(config)

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
                logging.exception("Vehicle detection failed")

    def add_frame(self, frame, roi_coords):
        self.frame_buffer.put((frame, roi_coords))

    def get_result(self):
        return self.result_buffer.get()

    def stop(self):
        self.processing = False
        if self.detection_thread:
            self.detection_thread.join()
