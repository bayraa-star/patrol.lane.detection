from dataclasses import dataclass, field
from pathlib import Path
import os

import yaml


_DEFAULT_CLASS_COLORS = (
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
)
_DEFAULT_ANNOTATION_TYPE_BY_CLASS = {
    "center": "polygon",
    "curve": "polygon",
    "dashed": "polygon",
    "patrol": "bbox",
    "solid": "polygon",
}
_VALID_ANNOTATION_TYPES = {"bbox", "polygon"}


@dataclass
class Config:
    @staticmethod
    def _parse_bool(value: str, default: bool) -> bool:
        if value is None:
            return default
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _parse_csv_floats(value: str, default: tuple[float, ...], expected_len: int) -> tuple[float, ...]:
        if not value:
            return default
        try:
            parts = tuple(float(item.strip()) for item in value.split(","))
        except (TypeError, ValueError):
            return default
        return parts if len(parts) == expected_len else default

    @staticmethod
    def _parse_csv_strings(value: str, default: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            return default
        parts = tuple(item.strip() for item in value.split(",") if item.strip())
        return parts or default

    @staticmethod
    def _parse_color(value: str, default: tuple[int, int, int]) -> tuple[int, int, int]:
        if not value:
            return default
        try:
            parts = tuple(int(item.strip()) for item in value.split(","))
        except (TypeError, ValueError):
            return default
        if len(parts) != 3:
            return default
        return tuple(max(0, min(channel, 255)) for channel in parts)

    @staticmethod
    def _normalize_annotation_types(
        class_names: list[str],
        raw_annotation_types,
    ) -> list[str]:
        if not raw_annotation_types:
            return []

        if isinstance(raw_annotation_types, dict):
            normalized = [
                str(raw_annotation_types.get(class_name, "")).strip().lower()
                for class_name in class_names
            ]
        elif isinstance(raw_annotation_types, list):
            normalized = [str(annotation_type).strip().lower() for annotation_type in raw_annotation_types]
        else:
            raise ValueError("annotation_types must be a list or mapping")

        if len(normalized) != len(class_names):
            raise ValueError("annotation_types must have the same number of entries as class names")

        invalid = [annotation_type for annotation_type in normalized if annotation_type not in _VALID_ANNOTATION_TYPES]
        if invalid:
            raise ValueError(
                f"Invalid annotation_types entries: {invalid}. Valid values: {sorted(_VALID_ANNOTATION_TYPES)}"
            )

        return normalized

    @staticmethod
    def _load_model_metadata(
        path: str,
        colors_are_rgb: bool,
    ) -> tuple[list[str], list[tuple[int, int, int]], list[str]]:
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Model config not found: {file_path}")

        with file_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}

        names = payload.get("names") or []
        if not isinstance(names, list) or not names:
            raise ValueError(f"No class names found in {file_path}")

        raw_colors = payload.get("colors") or []
        colors: list[tuple[int, int, int]] = []
        for color in raw_colors:
            if not isinstance(color, list) or len(color) < 3:
                continue
            triplet = tuple(max(0, min(int(channel), 255)) for channel in color[:3])
            if colors_are_rgb:
                triplet = (triplet[2], triplet[1], triplet[0])
            colors.append(triplet)

        if not colors:
            colors = list(_DEFAULT_CLASS_COLORS)

        while len(colors) < len(names):
            colors.extend(_DEFAULT_CLASS_COLORS)

        annotation_types = Config._normalize_annotation_types(
            [str(name) for name in names],
            payload.get("annotation_types"),
        )

        return [str(name) for name in names], colors[: len(names)], annotation_types

    @staticmethod
    def _resolve_class_annotation_types(
        class_names: list[str],
        metadata_types: list[str],
        override_types: tuple[str, ...],
    ) -> list[str]:
        if override_types:
            normalized = [annotation_type.strip().lower() for annotation_type in override_types]
            if len(normalized) != len(class_names):
                raise ValueError(
                    "CLASS_ANNOTATION_TYPES must have the same number of entries as class names"
                )
            invalid = [annotation_type for annotation_type in normalized if annotation_type not in _VALID_ANNOTATION_TYPES]
            if invalid:
                raise ValueError(
                    f"Invalid CLASS_ANNOTATION_TYPES entries: {invalid}. Valid values: {sorted(_VALID_ANNOTATION_TYPES)}"
                )
            return normalized

        if metadata_types:
            return metadata_types

        return [
            _DEFAULT_ANNOTATION_TYPE_BY_CLASS.get(class_name, "bbox")
            for class_name in class_names
        ]

    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(levelname)s - %(message)s"

    STREAM_PORT: int = int(os.getenv("STREAM_PORT", "8005"))
    VIDEO_FEED_JPEG_QUALITY: int = int(os.getenv("VIDEO_FEED_JPEG_QUALITY", "85"))
    STREAM_FRAME_QUEUE_SIZE: int = int(os.getenv("STREAM_FRAME_QUEUE_SIZE", "2"))

    RTSP_URL: str = os.getenv("RTSP_URL", "rtsp://localhost:8554/speed2")
    RECONNECT_ATTEMPTS: int = int(os.getenv("RECONNECT_ATTEMPTS", "10"))
    RECONNECT_DELAY: float = float(os.getenv("RECONNECT_DELAY", "5"))
    RTSP_PROTOCOLS: str = os.getenv("RTSP_PROTOCOLS", "tcp")
    RTSP_LATENCY_MS: int = int(os.getenv("RTSP_LATENCY_MS", "200"))
    RTSP_FPS: str = os.getenv("RTSP_FPS", "20/1")
    INTEL_DRM_DEVICE: str = os.getenv("INTEL_DRM_DEVICE", "/dev/dri/renderD128")
    AUTO_DETECT_INTEL_DRM: bool = _parse_bool.__func__(os.getenv("AUTO_DETECT_INTEL_DRM"), True)
    PREFER_INTEL_HW_DECODE: bool = _parse_bool.__func__(os.getenv("PREFER_INTEL_HW_DECODE"), True)
    ALLOW_CPU_FALLBACK: bool = _parse_bool.__func__(os.getenv("ALLOW_CPU_FALLBACK"), True)
    PREFER_FFMPEG_CPU_FALLBACK: bool = _parse_bool.__func__(os.getenv("PREFER_FFMPEG_CPU_FALLBACK"), True)
    LIBVA_DRIVER_NAME: str = os.getenv("LIBVA_DRIVER_NAME", "").strip()
    LIBVA_DRIVER_CANDIDATES: tuple[str, ...] = field(
        default_factory=lambda: Config._parse_csv_strings(
            os.getenv("LIBVA_DRIVER_CANDIDATES"),
            ("iHD", "i965"),
        )
    )
    FFMPEG_CAPTURE_OPTIONS: str = os.getenv("FFMPEG_CAPTURE_OPTIONS", "")
    GSTREAMER_HW_DECODERS: tuple[str, ...] = field(
        default_factory=lambda: Config._parse_csv_strings(
            os.getenv("GSTREAMER_HW_DECODERS"),
            ("vaapih264dec", "vah264dec"),
        )
    )
    GSTREAMER_SW_DECODERS: tuple[str, ...] = field(
        default_factory=lambda: Config._parse_csv_strings(
            os.getenv("GSTREAMER_SW_DECODERS"),
            ("avdec_h264",),
        )
    )

    MODEL_PATH: str = os.getenv("MODEL_PATH", "./models/lane/lane_detector.pt")
    MODEL_CONFIG_PATH: str = os.getenv("MODEL_CONFIG_PATH", "./models/lane/configs.yaml")
    MODEL_DEVICE: str = os.getenv("MODEL_DEVICE", "cpu")
    MODEL_IMAGE_SIZE: int = int(os.getenv("MODEL_IMAGE_SIZE", "640"))
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.05"))
    NMS_THRESHOLD: float = float(os.getenv("NMS_THRESHOLD", "0.45"))
    MAX_DETECTIONS: int = int(os.getenv("MAX_DETECTIONS", "100"))

    DETECTION_ROI: tuple[float, float, float, float] = field(
        default_factory=lambda: Config._parse_csv_floats(
            os.getenv("DETECTION_ROI"),
            (0.0, 0.0, 1.0, 1.0),
            4,
        )
    )
    RAW_FRAME_QUEUE_SIZE: int = int(os.getenv("RAW_FRAME_QUEUE_SIZE", "1"))
    DETECTOR_INPUT_QUEUE_SIZE: int = int(os.getenv("DETECTOR_INPUT_QUEUE_SIZE", "1"))
    DETECTOR_RESULT_QUEUE_SIZE: int = int(os.getenv("DETECTOR_RESULT_QUEUE_SIZE", "1"))
    DETECTOR_POLL_TIMEOUT_S: float = float(os.getenv("DETECTOR_POLL_TIMEOUT_S", "0.05"))
    SKIP_FRAMES: int = int(os.getenv("SKIP_FRAMES", "1"))
    MAX_FPS: int = int(os.getenv("MAX_FPS", "20"))
    RESULT_STALE_SECONDS: float = float(os.getenv("RESULT_STALE_SECONDS", "0.35"))
    STATS_WINDOW_SIZE: int = int(os.getenv("STATS_WINDOW_SIZE", "100"))

    DRAW_ROI: bool = _parse_bool.__func__(os.getenv("DRAW_ROI"), True)
    ROI_COLOR: tuple[int, int, int] = field(
        default_factory=lambda: Config._parse_color(os.getenv("ROI_COLOR"), (255, 0, 0))
    )
    BOX_THICKNESS: int = int(os.getenv("BOX_THICKNESS", "2"))
    TEXT_SCALE: float = float(os.getenv("TEXT_SCALE", "0.6"))
    TEXT_THICKNESS: int = int(os.getenv("TEXT_THICKNESS", "2"))
    LABEL_COLORS_ARE_RGB: bool = _parse_bool.__func__(os.getenv("LABEL_COLORS_ARE_RGB"), True)
    CLASS_ANNOTATION_TYPES_OVERRIDE: tuple[str, ...] = field(
        default_factory=lambda: Config._parse_csv_strings(
            os.getenv("CLASS_ANNOTATION_TYPES"),
            (),
        )
    )

    CLASS_NAMES: list[str] = field(init=False)
    CLASS_COLORS: list[tuple[int, int, int]] = field(init=False)
    CLASS_ANNOTATION_TYPES: list[str] = field(init=False)
    CLASS_METADATA: dict[str, dict[str, object]] = field(init=False)

    def __post_init__(self):
        self.SKIP_FRAMES = max(1, self.SKIP_FRAMES)
        self.MAX_FPS = max(1, self.MAX_FPS)
        self.STATS_WINDOW_SIZE = max(1, self.STATS_WINDOW_SIZE)
        self.STREAM_FRAME_QUEUE_SIZE = max(1, self.STREAM_FRAME_QUEUE_SIZE)
        self.RAW_FRAME_QUEUE_SIZE = max(1, self.RAW_FRAME_QUEUE_SIZE)
        self.DETECTOR_INPUT_QUEUE_SIZE = max(1, self.DETECTOR_INPUT_QUEUE_SIZE)
        self.DETECTOR_RESULT_QUEUE_SIZE = max(1, self.DETECTOR_RESULT_QUEUE_SIZE)
        self.DETECTOR_POLL_TIMEOUT_S = max(0.001, self.DETECTOR_POLL_TIMEOUT_S)
        self.VIDEO_FEED_JPEG_QUALITY = max(1, min(self.VIDEO_FEED_JPEG_QUALITY, 100))
        self.LIBVA_DRIVER_NAME = self.LIBVA_DRIVER_NAME.strip()

        libva_driver_candidates: list[str] = []
        if self.LIBVA_DRIVER_NAME:
            libva_driver_candidates.append(self.LIBVA_DRIVER_NAME)
        for driver_name in self.LIBVA_DRIVER_CANDIDATES:
            normalized_name = driver_name.strip()
            if normalized_name and normalized_name not in libva_driver_candidates:
                libva_driver_candidates.append(normalized_name)
        self.LIBVA_DRIVER_CANDIDATES = tuple(libva_driver_candidates or ["iHD", "i965"])

        if not Path(self.MODEL_PATH).is_file():
            raise FileNotFoundError(f"Model not found: {self.MODEL_PATH}")

        x1, y1, x2, y2 = self.DETECTION_ROI
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            raise ValueError(
                "DETECTION_ROI must be normalized values in [0, 1] with x1 < x2 and y1 < y2"
            )

        self.CLASS_NAMES, self.CLASS_COLORS, metadata_annotation_types = self._load_model_metadata(
            self.MODEL_CONFIG_PATH,
            self.LABEL_COLORS_ARE_RGB,
        )
        self.CLASS_ANNOTATION_TYPES = self._resolve_class_annotation_types(
            self.CLASS_NAMES,
            metadata_annotation_types,
            self.CLASS_ANNOTATION_TYPES_OVERRIDE,
        )
        self.CLASS_METADATA = {
            class_name: {
                "class_id": index,
                "color": self.CLASS_COLORS[index],
                "annotation_type": self.CLASS_ANNOTATION_TYPES[index],
            }
            for index, class_name in enumerate(self.CLASS_NAMES)
        }
