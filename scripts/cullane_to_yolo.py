#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import shutil
from pathlib import Path

import yaml


DEFAULT_IMAGE_WIDTH = 1640
DEFAULT_IMAGE_HEIGHT = 590
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert CULane .lines annotations into a YOLO lane dataset.",
    )
    parser.add_argument(
        "--culane-root",
        required=True,
        help="Path to the original CULane root directory containing driver_* and list/.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Directory where the converted YOLO dataset will be written.",
    )
    parser.add_argument(
        "--task",
        choices=("segment", "detect"),
        default="segment",
        help="Write YOLO segmentation polygons or detection boxes.",
    )
    parser.add_argument(
        "--lane-width",
        type=int,
        default=16,
        help="Pixel width used to turn CULane polylines into thin segmentation polygons.",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=DEFAULT_IMAGE_WIDTH,
        help="Fallback image width when an image cannot be read.",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=DEFAULT_IMAGE_HEIGHT,
        help="Fallback image height when an image cannot be read.",
    )
    parser.add_argument(
        "--image-mode",
        choices=("symlink", "copy", "none"),
        default="symlink",
        help="How to place images in the converted dataset.",
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Also convert CULane test lists when present.",
    )
    parser.add_argument(
        "--overwrite-labels",
        action="store_true",
        help="Rewrite existing label files.",
    )
    return parser


def _read_split_entries(culane_root: Path, split_name: str) -> list[Path]:
    list_file = culane_root / "list" / f"{split_name}_gt.txt"
    if split_name == "test" and not list_file.is_file():
        test_split_dir = culane_root / "list" / "test_split"
        if not test_split_dir.is_dir():
            return []
        entries: list[Path] = []
        for split_file in sorted(test_split_dir.glob("*.txt")):
            entries.extend(_read_image_paths(culane_root, split_file))
        return entries

    if not list_file.is_file():
        return []
    return _read_image_paths(culane_root, list_file)


def _read_image_paths(culane_root: Path, list_file: Path) -> list[Path]:
    image_paths: list[Path] = []
    with list_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.strip().split()
            if not fields:
                continue
            image_paths.append(_resolve_culane_path(culane_root, fields[0]))
    return image_paths


def _resolve_culane_path(culane_root: Path, raw_path: str) -> Path:
    relative_path = raw_path.lstrip("/")
    return (culane_root / relative_path).resolve()


def _relative_image_path(culane_root: Path, image_path: Path) -> Path:
    try:
        relative_path = image_path.resolve().relative_to(culane_root.resolve())
    except ValueError:
        relative_path = Path(image_path.name)
    return relative_path


def _get_image_shape(image_path: Path, fallback_width: int, fallback_height: int) -> tuple[int, int]:
    try:
        import cv2
    except ImportError:
        return fallback_width, fallback_height

    image = cv2.imread(str(image_path))
    if image is None:
        return fallback_width, fallback_height
    height, width = image.shape[:2]
    return width, height


def _read_lanes(annotation_path: Path) -> list[list[tuple[float, float]]]:
    if not annotation_path.is_file():
        return []

    lanes: list[list[tuple[float, float]]] = []
    with annotation_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            values = line.strip().split()
            if len(values) < 4 or len(values) % 2 != 0:
                continue

            points: list[tuple[float, float]] = []
            for index in range(0, len(values), 2):
                try:
                    x_value = float(values[index])
                    y_value = float(values[index + 1])
                except ValueError:
                    continue
                if x_value < 0 or y_value < 0:
                    continue
                points.append((x_value, y_value))

            if len(points) >= 2:
                lanes.append(points)
    return lanes


def _clip_points(points: list[tuple[float, float]], width: int, height: int) -> list[tuple[float, float]]:
    clipped = []
    for x_value, y_value in points:
        x_value = min(max(x_value, 0.0), float(width - 1))
        y_value = min(max(y_value, 0.0), float(height - 1))
        clipped.append((x_value, y_value))
    return clipped


def _bbox_label(points: list[tuple[float, float]], width: int, height: int) -> str | None:
    clipped = _clip_points(points, width, height)
    xs = [point[0] for point in clipped]
    ys = [point[1] for point in clipped]
    if not xs or not ys:
        return None

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    if x_max <= x_min or y_max <= y_min:
        return None

    x_center = ((x_min + x_max) / 2.0) / width
    y_center = ((y_min + y_max) / 2.0) / height
    box_width = (x_max - x_min) / width
    box_height = (y_max - y_min) / height
    return f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"


def _segment_label(points: list[tuple[float, float]], width: int, height: int, lane_width: int) -> str | None:
    clipped = _clip_points(points, width, height)
    if len(clipped) < 2:
        return None

    half_width = max(1.0, float(lane_width) / 2.0)
    left_side = []
    right_side = []
    for index, (x_value, y_value) in enumerate(clipped):
        if index == 0:
            next_x, next_y = clipped[index + 1]
            dx = next_x - x_value
            dy = next_y - y_value
        elif index == len(clipped) - 1:
            previous_x, previous_y = clipped[index - 1]
            dx = x_value - previous_x
            dy = y_value - previous_y
        else:
            previous_x, previous_y = clipped[index - 1]
            next_x, next_y = clipped[index + 1]
            dx = next_x - previous_x
            dy = next_y - previous_y

        length = math.hypot(dx, dy)
        if length <= 1e-6:
            continue
        normal_x = -dy / length
        normal_y = dx / length
        left_side.append((x_value + normal_x * half_width, y_value + normal_y * half_width))
        right_side.append((x_value - normal_x * half_width, y_value - normal_y * half_width))

    polygon = left_side + list(reversed(right_side))
    if len(polygon) < 3:
        return None

    normalized_points = []
    for x_value, y_value in polygon:
        normalized_points.append(f"{min(max(float(x_value) / width, 0.0), 1.0):.6f}")
        normalized_points.append(f"{min(max(float(y_value) / height, 0.0), 1.0):.6f}")
    return "0 " + " ".join(normalized_points)


def _place_image(source_path: Path, output_path: Path, image_mode: str) -> None:
    if image_mode == "none":
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() or output_path.is_symlink():
        return

    if image_mode == "copy":
        shutil.copy2(source_path, output_path)
        return

    relative_source = os.path.relpath(source_path, output_path.parent)
    output_path.symlink_to(relative_source)


def _convert_split(
    culane_root: Path,
    output_root: Path,
    split_name: str,
    image_paths: list[Path],
    task: str,
    lane_width: int,
    fallback_width: int,
    fallback_height: int,
    image_mode: str,
    overwrite_labels: bool,
) -> tuple[int, int]:
    converted_images = 0
    converted_labels = 0

    for image_path in image_paths:
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if not image_path.is_file():
            continue

        relative_path = _relative_image_path(culane_root, image_path)
        output_image_path = output_root / "images" / split_name / relative_path
        output_label_path = (output_root / "labels" / split_name / relative_path).with_suffix(".txt")

        _place_image(image_path, output_image_path, image_mode)
        converted_images += 1

        if output_label_path.exists() and not overwrite_labels:
            converted_labels += 1
            continue

        lanes = _read_lanes(image_path.with_suffix(".lines"))
        width, height = _get_image_shape(image_path, fallback_width, fallback_height)
        labels = []
        for lane in lanes:
            if task == "segment":
                label = _segment_label(lane, width, height, lane_width)
            else:
                label = _bbox_label(lane, width, height)
            if label:
                labels.append(label)

        output_label_path.parent.mkdir(parents=True, exist_ok=True)
        output_label_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
        converted_labels += 1

    return converted_images, converted_labels


def _write_yaml(output_root: Path, task: str) -> Path:
    yaml_path = output_root / "cullane.yaml"
    payload = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": ["lane"],
        "colors": [[0, 255, 0]],
        "annotation_types": ["polygon" if task == "segment" else "bbox"],
        "dataset": "CULane",
        "task": task,
    }
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return yaml_path


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    culane_root = Path(args.culane_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not culane_root.is_dir():
        parser.error(f"CULane root not found: {culane_root}")

    split_names = ["train", "val"]
    if args.include_test:
        split_names.append("test")

    output_root.mkdir(parents=True, exist_ok=True)
    totals = {}
    for split_name in split_names:
        image_paths = _read_split_entries(culane_root, split_name)
        totals[split_name] = _convert_split(
            culane_root=culane_root,
            output_root=output_root,
            split_name=split_name,
            image_paths=image_paths,
            task=args.task,
            lane_width=args.lane_width,
            fallback_width=args.image_width,
            fallback_height=args.image_height,
            image_mode=args.image_mode,
            overwrite_labels=args.overwrite_labels,
        )

    yaml_path = _write_yaml(output_root, args.task)
    for split_name, (image_count, label_count) in totals.items():
        print(f"{split_name}: {image_count} images, {label_count} labels")
    print(f"Wrote dataset config: {yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
