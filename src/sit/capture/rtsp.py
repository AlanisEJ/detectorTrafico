from __future__ import annotations

import math
import os
import queue
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Iterator

from sit.config import CameraConfig
from sit.domain.types import Frame, StreamBoundary


class CaptureStream:
    """Cola de dos mensajes: RTSP conserva frames recientes; video conserva todos."""

    def __init__(self, camera: CameraConfig, stop: threading.Event, *, cv2_module=None):
        self.camera, self.stop = camera, stop
        self.local_stop = threading.Event()
        self.done = threading.Event()
        self.messages: queue.Queue[Frame | StreamBoundary] = queue.Queue(maxsize=2)
        self.cv2 = cv2_module
        self.error_type: str | None = None
        self.thread = threading.Thread(target=self._produce, name=f"capture-{camera.camera_id}", daemon=True)

    def stopping(self) -> bool:
        return self.stop.is_set() or self.local_stop.is_set()

    def _wait(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while not self.stopping() and time.monotonic() < deadline:
            self.local_stop.wait(min(0.1, max(0, deadline - time.monotonic())))

    def _put(self, message: Frame | StreamBoundary, *, latest: bool = False) -> None:
        while not self.stopping():
            try:
                self.messages.put(message, timeout=0.05)
                return
            except queue.Full:
                if latest:
                    try:
                        self.messages.get_nowait()
                    except queue.Empty:
                        pass

    def _open(self, source: str):
        cv2 = self.cv2
        if self.camera.kind == "rtsp":
            return cv2.VideoCapture(source, cv2.CAP_FFMPEG, [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.camera.open_timeout_ms,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.camera.read_timeout_ms])
        return cv2.VideoCapture(source)

    def _produce(self) -> None:
        cap = None
        try:
            # Evitar que FFmpeg/OpenCV impriman la URL con credenciales en stderr.
            os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
            os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
            if self.cv2 is None:
                import cv2
                self.cv2 = cv2
            if hasattr(self.cv2, "setLogLevel"):
                self.cv2.setLogLevel(0)
            source = self.camera.resolve_source()
            started = time.monotonic()
            wall_start = self.camera.start_at or datetime.now(timezone.utc)
            sequence, epoch = 0, 0
            while not self.stopping():
                cap = self._open(source)
                if not cap.isOpened():
                    cap.release()
                    cap = None
                    if self.camera.kind == "video":
                        raise ValueError("No se pudo abrir el video")
                    self._put(StreamBoundary("disconnect"))
                    self._wait(self.camera.reconnect_delay_s)
                    continue
                epoch += 1
                cap.set(self.cv2.CAP_PROP_BUFFERSIZE, self.camera.buffer_size)
                fps = self.camera.fps_override or cap.get(self.cv2.CAP_PROP_FPS)
                if self.camera.kind == "video" and (not math.isfinite(fps) or fps <= 0):
                    raise ValueError("El video necesita FPS válidos o source.fps")
                index = 0
                while not self.stopping():
                    ok, image = cap.read()
                    if not ok:
                        if self.camera.kind == "video":
                            self._put(StreamBoundary("eof"))
                            return
                        self._put(StreamBoundary("disconnect"))
                        break
                    index += 1
                    sequence += 1
                    if (index - 1) % self.camera.frame_skip:
                        continue
                    elapsed = (index - 1) / fps if self.camera.kind == "video" else time.monotonic() - started
                    timestamp = wall_start + timedelta(seconds=elapsed) if self.camera.kind == "video" else datetime.now(timezone.utc)
                    image = self.cv2.resize(image, self.camera.resolution)
                    packet = Frame(self.camera.camera_id, sequence, elapsed, timestamp, epoch, image)
                    self._put(packet, latest=self.camera.kind == "rtsp")
                cap.release()
                cap = None
                self._wait(self.camera.reconnect_delay_s)
        except Exception as error:
            # No transportar el texto de un error de backend que podría incluir la URL.
            self.error_type = type(error).__name__
            self._put(StreamBoundary("error"))
        finally:
            if cap is not None:
                cap.release()
            self.done.set()

    def __enter__(self) -> CaptureStream:
        self.thread.start()
        return self

    def __iter__(self) -> Iterator[Frame | StreamBoundary]:
        while not self.stopping():
            try:
                yield self.messages.get(timeout=0.1)
            except queue.Empty:
                if self.done.is_set():
                    return

    def __exit__(self, *_args) -> None:
        self.local_stop.set()
        timeout = (self.camera.open_timeout_ms + self.camera.read_timeout_ms) / 1000 + 1
        self.thread.join(timeout=timeout)
        if self.thread.is_alive():
            raise RuntimeError("El backend no respetó el timeout de captura")
