# ============================
# Stage 1: Build OpenCV + GStreamer (+ Python cv2)
# ============================
FROM ubuntu:22.04 AS opencv-builder
ENV DEBIAN_FRONTEND=noninteractive TZ=UTC
ARG OPENCV_VERSION=4.10.0

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
      build-essential cmake git pkg-config \
      python3 python3-dev python3-pip python3-venv python3-numpy \
      libgtk-3-dev libtbb-dev libeigen3-dev \
      libjpeg-dev libpng-dev libtiff-dev libopenexr-dev \
      libavcodec-dev libavformat-dev libswscale-dev \
      libxvidcore-dev libx264-dev libv4l-dev libdc1394-dev \
      libopenblas-dev liblapack-dev gfortran \
      libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev \
      gstreamer1.0-tools gstreamer1.0-libav \
      gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
      gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch ${OPENCV_VERSION} https://github.com/opencv/opencv.git /opencv
WORKDIR /opencv/build

RUN PYV=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")') && \
    NPYI=$(python3 -c 'import numpy as np; print(np.get_include())') && \
    OPY="/usr/local/lib/python${PYV}/dist-packages" && \
    cmake -D CMAKE_BUILD_TYPE=Release \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D WITH_GSTREAMER=ON \
          -D WITH_FFMPEG=ON \
          -D WITH_TBB=ON \
          -D WITH_IPP=ON \
          -D WITH_OPENMP=ON \
          -D WITH_V4L=ON \
          -D WITH_LIBV4L=ON \
          -D BUILD_EXAMPLES=OFF \
          -D BUILD_TESTS=OFF \
          -D BUILD_PERF_TESTS=OFF \
          -D OPENCV_GENERATE_PKGCONFIG=ON \
          -D BUILD_opencv_python3=ON \
          -D PYTHON3_EXECUTABLE=/usr/bin/python3 \
          -D OPENCV_PYTHON3_INSTALL_PATH="$OPY" \
          -D PYTHON3_NUMPY_INCLUDE_DIRS="$NPYI" \
          .. && \
    make -j"$(nproc)" && make install && ldconfig

RUN python3 - <<'PY'
import cv2, re
info = cv2.getBuildInformation()
m = re.search(r'GStreamer:\s+(YES|NO)', info)
status = m.group(1) if m else "MISSING"
assert status == "YES", f"OpenCV was not built with GStreamer (status={status})"
print("Builder OK: GStreamer:", status, "cv2 at", cv2.__file__)
PY


# ============================
# Stage 2: Runtime (slim)
# ============================
FROM ubuntu:22.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv python3-numpy \
      libgtk-3-0 \
      libtbb12 libtbbmalloc2 \
      libjpeg-turbo8 libpng16-16 libtiff5 libopenexr25 \
      libavcodec58 libavformat58 libswscale5 \
      libxvidcore4 libx264-163 libv4l-0 libdc1394-25 \
      libopenblas0 liblapack3 \
      gstreamer1.0-tools gstreamer1.0-libav \
      gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
      gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
      gstreamer1.0-vaapi vainfo intel-media-va-driver i965-va-driver \
    && rm -rf /var/lib/apt/lists/*

COPY --from=opencv-builder /usr/local/ /usr/local/

RUN python3 - <<'PY'
import sys
from pathlib import Path

root = Path(f"/usr/local/lib/python{sys.version_info[0]}.{sys.version_info[1]}/dist-packages")
candidates = sorted(root.glob("cv2/python-*"))
if not candidates:
    raise SystemExit(f"No OpenCV Python extension directory found under {root}")

pth_file = root / "opencv-local.pth"
pth_file.write_text(str(candidates[0]) + "\n", encoding="utf-8")
print("Configured OpenCV Python path:", candidates[0])
PY

COPY requirements.txt /tmp/requirements.txt

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r /tmp/requirements.txt && \
    (pip3 uninstall -y opencv-python opencv-contrib-python opencv-python-headless || true)

RUN python3 - <<'PY'
import sys, numpy as np, cv2, re, yaml
print("Python:", sys.version)
print("NumPy:", np.__version__)
print("cv2  :", cv2.__file__)
print("yaml :", yaml.__file__)
assert hasattr(cv2, "getBuildInformation"), "cv2 loaded without OpenCV Python bindings"
print(re.search(r'GStreamer:\s+(YES|NO)', cv2.getBuildInformation()).group(0))
PY

WORKDIR /app
COPY . /app
EXPOSE 8005
CMD ["python3", "main.py"]
