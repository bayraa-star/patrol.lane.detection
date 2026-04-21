#!/usr/bin/env python3
import argparse
import logging

import cv2

from config.config import Config
from utils.lane_detector import LaneDetector, draw_detections, draw_roi, get_roi_coordinates


def main():
    config = Config()
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
        format=config.LOG_FORMAT,
    )

    parser = argparse.ArgumentParser(description="Run lane detection on a still image")
    parser.add_argument("input_image", type=str, help="Path to the input image")
    parser.add_argument(
        "--output-image",
        type=str,
        default="output.jpg",
        help="Path to save the annotated output image",
    )
    parser.add_argument(
        "--full-frame",
        action="store_true",
        help="Run inference on the full frame instead of the configured ROI",
    )
    args = parser.parse_args()

    image = cv2.imread(args.input_image)
    if image is None:
        logging.error("Failed to load image: %s", args.input_image)
        return

    detector = LaneDetector(config)
    roi_coords = None if args.full_frame else get_roi_coordinates(image.shape, config.DETECTION_ROI)
    detections = detector.detect(image, roi_coords)

    annotated = image.copy()
    if roi_coords is not None and config.DRAW_ROI:
        draw_roi(annotated, roi_coords, config.ROI_COLOR, config.BOX_THICKNESS)

    draw_detections(
        annotated,
        detections,
        config.CLASS_NAMES,
        config.CLASS_COLORS,
        roi_coords=roi_coords,
        line_thickness=config.BOX_THICKNESS,
        text_scale=config.TEXT_SCALE,
        text_thickness=config.TEXT_THICKNESS,
    )

    cv2.imwrite(args.output_image, annotated)
    logging.info("Saved %s detection(s) to %s", len(detections), args.output_image)


if __name__ == "__main__":
    main()
