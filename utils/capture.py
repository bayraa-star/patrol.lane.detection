# utils/capture.py
import time
import logging
from queue import Full, Empty

def capture_thread_func(rtsp_conn, raw_frame_queue, stop_event):
    consecutive_failures = 0
    max_consecutive_failures = 200  # ~6s @ 30fps
    idle_sleep = 0.02

    while not stop_event.is_set():
        if not rtsp_conn.is_connected:
            time.sleep(0.05)
            continue

        ok, frame = rtsp_conn.read()
        if not ok or frame is None:
            consecutive_failures += 1
            if consecutive_failures % 30 == 0:
                logging.warning("⚠️ Frame read failed in capture thread. Consecutive: %d", consecutive_failures)
            if consecutive_failures >= max_consecutive_failures:
                logging.warning("⛔ No frames for a while; marking disconnected")
                rtsp_conn.is_connected = False
            time.sleep(idle_sleep)
            continue

        consecutive_failures = 0
        try:
            raw_frame_queue.put(frame, block=False)
        except Full:
            # Keep the freshest frame for low-latency pipelines.
            try:
                raw_frame_queue.get_nowait()
            except Empty:
                pass
            try:
                raw_frame_queue.put(frame, block=False)
            except Full:
                pass
