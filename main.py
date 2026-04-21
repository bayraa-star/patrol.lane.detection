#!/usr/bin/env python3
import logging
import threading
import time
from queue import Empty, Full, Queue
from urllib.parse import urlparse, urlunparse

import cv2
import numpy as np

from config.config import Config
from service.flask_server import frame_queue, run_flask
from utils.capture import capture_thread_func
from utils.lane_detector import AsyncLaneDetector, RTSPConnection, draw_detections, draw_roi, get_roi_coordinates


def configure_logging(config):
    log_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format=config.LOG_FORMAT, force=True)


def redact_rtsp(url):
    try:
        parsed = urlparse(url)
        if parsed.username is None:
            return url

        host_part = parsed.hostname or ""
        if parsed.port:
            host_part += f":{parsed.port}"

        if parsed.password is None:
            netloc = f"{parsed.username}@{host_part}"
        else:
            netloc = f"{parsed.username}:****@{host_part}"

        return urlunparse((parsed.scheme, netloc, parsed.path, parsed.params, parsed.query, parsed.fragment))
    except Exception:
        return "<invalid-rtsp-url>"


def log_rtsp_details(url):
    logging.info("RTSP URL: %s", redact_rtsp(url))
    parsed = urlparse(url)
    if parsed.hostname:
        host = parsed.hostname if parsed.port is None else f"{parsed.hostname}:{parsed.port}"
        logging.info("RTSP host: %s", host)


def connect_with_retries(conn, retries, retry_delay):
    attempt = 0
    while attempt < retries:
        if conn.connect():
            if conn.cap:
                conn.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if conn.active_decode_mode == "hardware":
                logging.info(
                    "Connected RTSP stream using Intel hardware decoder '%s' on %s",
                    conn.active_decoder,
                    conn.active_drm_device,
                )
            else:
                logging.info(
                    "Connected RTSP stream using CPU decoder '%s'",
                    conn.active_decoder,
                )
            return True

        attempt += 1
        logging.warning("RTSP connection failed. Retry %s/%s", attempt, retries)
        time.sleep(retry_delay)

    logging.error("Failed to establish the RTSP stream with any configured decoder")
    return False


def drain_latest_result(detector):
    latest = None
    while True:
        result = detector.get_result()
        if result is None:
            break
        latest = result
    return latest


def enqueue_frame(frame):
    try:
        while frame_queue.qsize() > 0:
            frame_queue.get_nowait()
        frame_queue.put(frame, block=False)
    except Full:
        logging.warning("Video feed queue full, dropping frame")


def main():
    config = Config()
    configure_logging(config)

    logging.info("Starting lane detection service")
    logging.info("Lane classes: %s", ", ".join(config.CLASS_NAMES))
    logging.info("OpenCV has GStreamer support: %s", "GStreamer:                   YES" in cv2.getBuildInformation())
    log_rtsp_details(config.RTSP_URL)

    flask_thread = threading.Thread(target=run_flask, args=(config,), daemon=True)
    flask_thread.start()
    logging.info("Streaming server started on port %s", config.STREAM_PORT)

    rtsp_conn = RTSPConnection(
        rtsp_url=config.RTSP_URL,
        max_retries=config.RECONNECT_ATTEMPTS,
        retry_delay=config.RECONNECT_DELAY,
        connection_timeout=15,
        config=config,
        fps=config.RTSP_FPS,
    )

    if not connect_with_retries(rtsp_conn, config.RECONNECT_ATTEMPTS, config.RECONNECT_DELAY):
        return

    detector = AsyncLaneDetector(config)
    detector.start_processing()
    logging.info("Model task: %s", detector.detector.model_task)
    logging.info(
        "Class annotation types: %s",
        ", ".join(
            f"{class_name}={annotation_type}"
            for class_name, annotation_type in zip(config.CLASS_NAMES, config.CLASS_ANNOTATION_TYPES)
        ),
    )

    raw_frame_queue = Queue(maxsize=config.RAW_FRAME_QUEUE_SIZE)
    stop_event = threading.Event()
    capture_thread = threading.Thread(
        target=capture_thread_func,
        args=(rtsp_conn, raw_frame_queue, stop_event),
        daemon=True,
    )
    capture_thread.start()

    frame_count = 0
    last_loop_time = time.time()
    fps_limit_delay = 1.0 / max(config.MAX_FPS, 1)
    process_times = []
    connection_stats = {
        "reconnections": 0,
        "frames_processed": 0,
        "uptime": time.time(),
    }

    last_detections = []
    last_result_ts = 0.0

    try:
        while True:
            if not rtsp_conn.check_health():
                logging.warning("RTSP stream health check failed")
                rtsp_conn.is_connected = False

            if not rtsp_conn.is_connected:
                logging.info("Attempting to reconnect RTSP stream with Intel hardware decode")
                connection_stats["reconnections"] += 1
                connection_stats["uptime"] = time.time()
                if not rtsp_conn.reconnect():
                    logging.error("RTSP reconnection failed and CPU fallback is disabled. Exiting.")
                    break
                if rtsp_conn.cap:
                    rtsp_conn.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                logging.info("RTSP stream reconnected")

            try:
                frame = raw_frame_queue.get(timeout=0.5)
            except Empty:
                continue

            frame_count += 1
            connection_stats["frames_processed"] = frame_count
            roi_coords = get_roi_coordinates(frame.shape, config.DETECTION_ROI)

            if frame_count % config.SKIP_FRAMES == 0:
                detector.add_frame(frame.copy(), roi_coords)

            latest_result = drain_latest_result(detector)
            if latest_result is not None:
                _, detections, _, process_time, result_ts = latest_result
                last_detections = detections
                last_result_ts = result_ts
                process_times.append(process_time)
                if len(process_times) > config.STATS_WINDOW_SIZE:
                    process_times.pop(0)
                if detections:
                    logging.info("Frame %s: found %s lane detection(s)", frame_count, len(detections))

            processed_frame = frame.copy()
            if config.DRAW_ROI:
                draw_roi(processed_frame, roi_coords, config.ROI_COLOR, config.BOX_THICKNESS)

            if time.time() - last_result_ts <= config.RESULT_STALE_SECONDS:
                detections_to_draw = last_detections
            else:
                detections_to_draw = []

            if detections_to_draw:
                draw_detections(
                    processed_frame,
                    detections_to_draw,
                    config.CLASS_NAMES,
                    config.CLASS_COLORS,
                    roi_coords=roi_coords,
                    line_thickness=config.BOX_THICKNESS,
                    text_scale=config.TEXT_SCALE,
                    text_thickness=config.TEXT_THICKNESS,
                )

            enqueue_frame(processed_frame)

            elapsed = time.time() - last_loop_time
            target_delay = fps_limit_delay - elapsed
            if target_delay > 0:
                time.sleep(min(target_delay, 0.005))
            last_loop_time = time.time()

    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    except Exception:
        logging.exception("Unexpected runtime error")
    finally:
        logging.info("Cleaning up")
        stop_event.set()
        capture_thread.join()
        detector.stop()
        rtsp_conn.close()
        cv2.destroyAllWindows()

        if process_times:
            logging.info("Average detection time: %.3fs", float(np.mean(process_times)))
            logging.info("Min detection time: %.3fs", float(np.min(process_times)))
            logging.info("Max detection time: %.3fs", float(np.max(process_times)))

        logging.info("Total frames processed: %s", connection_stats["frames_processed"])
        logging.info("Reconnections: %s", connection_stats["reconnections"])
        logging.info("Stream uptime: %.0fs", time.time() - connection_stats["uptime"])


if __name__ == "__main__":
    main()
