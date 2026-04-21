from flask import Flask, Response
from flask_cors import CORS
from queue import Empty, Queue
import cv2

from config.config import Config

app = Flask(__name__)
CORS(app)
frame_queue = Queue(maxsize=Config.STREAM_FRAME_QUEUE_SIZE)
_jpeg_quality = 70


def configure_stream(config):
    global _jpeg_quality
    _jpeg_quality = config.VIDEO_FEED_JPEG_QUALITY


def generate_frames():
    while True:
        frame = frame_queue.get()

        try:
            while True:
                frame = frame_queue.get_nowait()
        except Empty:
            pass

        if frame is None:
            continue

        ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _jpeg_quality])
        if ret:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


def run_flask(config):
    configure_stream(config)
    app.run(host="0.0.0.0", port=config.STREAM_PORT, threaded=True, use_reloader=False)
