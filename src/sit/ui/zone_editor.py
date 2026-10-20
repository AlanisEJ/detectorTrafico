from __future__ import annotations

import queue
import threading

from sit.capture.rtsp import CaptureStream
from sit.config import CameraConfig
from sit.domain.types import Frame, StreamBoundary
from sit.domain.zones import Zone, load_zone, save_zone


def edit_zone(camera: CameraConfig, zone_index: int = 0) -> None:
    import cv2
    import numpy as np

    original = camera.zones[zone_index]
    path = camera.zone_paths[zone_index]
    points = [list(point) for point in original.points]
    selected = None
    title = f"SIT {camera.camera_id} | S guardar L cargar C limpiar ESC salir"
    stop = threading.Event()

    def mouse(event, x, y, _flags, _param):
        nonlocal selected
        if event == cv2.EVENT_LBUTTONDOWN:
            selected = next((i for i, point in enumerate(points)
                             if abs(point[0] - x) < 15 and abs(point[1] - y) < 15), None)
            if selected is None:
                points.append([x, y])
        elif event == cv2.EVENT_MOUSEMOVE and selected is not None:
            points[selected] = [x, y]
        elif event == cv2.EVENT_LBUTTONUP:
            selected = None

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, mouse)
    image = np.zeros((camera.resolution[1], camera.resolution[0], 3), dtype=np.uint8)
    status = "Conectando"
    try:
        with CaptureStream(camera, stop, cv2_module=cv2) as stream:
            while not stop.is_set():
                try:
                    message = stream.messages.get(timeout=0.05)
                    if isinstance(message, Frame):
                        image = message.image
                        status = "En vivo" if camera.kind == "rtsp" else "Video"
                    elif isinstance(message, StreamBoundary):
                        status = message.reason
                        if message.reason == "error":
                            raise RuntimeError("Falló la captura del editor")
                except queue.Empty:
                    pass
                canvas = image.copy()
                if len(points) >= 2:
                    cv2.polylines(canvas, [np.array(points, np.int32)], len(points) >= 3, (0, 200, 255), 2)
                for point in points:
                    cv2.circle(canvas, tuple(map(int, point)), 6, (0, 255, 255), -1)
                cv2.putText(canvas, status, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imshow(title, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord("c"):
                    points.clear()
                    selected = None
                elif key == ord("l"):
                    points[:] = [list(p) for p in load_zone(path).points]
                    selected = None
                    status = "Zona cargada"
                elif key == ord("s"):
                    try:
                        zone = Zone(original.zone_id, camera.camera_id, camera.resolution,
                                    tuple(tuple(p) for p in points))
                        save_zone(path, zone)
                        status = "Zona guardada; reinicie la captura para aplicarla"
                    except ValueError as error:
                        status = str(error)
    finally:
        stop.set()
        cv2.destroyAllWindows()
