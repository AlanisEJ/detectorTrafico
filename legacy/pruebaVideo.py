import cv2
import numpy as np
from ultralytics import YOLO
import threading
import torch
import time
from datetime import datetime

import os

rtsp = os.environ["SIT_CAM_1_RTSP"]

device = 0 if torch.cuda.is_available() else "cpu"

model = YOLO("yolov8m-seg.pt")
model.to(device)

if device != "cpu":
    model.model.half()

vehicle_classes = [2, 3, 5, 7]
vehicle_names = {2:"Auto",3:"Moto",5:"Bus",7:"Camion"}

cap = cv2.VideoCapture(rtsp)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

cv2.namedWindow("Deteccion de Vehiculos", cv2.WINDOW_NORMAL)

zona_trafico = []
punto_seleccionado = None

vehiculos_en_zona = {}
historial_centros = {}
entrada_zona = {}
vehiculos_previos = set()
kalman_filters = {}
tipo_vehiculo = {}   # NUEVO: guardar tipo real

frame_for_detection = None
lock = threading.Lock()

LOG_FILE = "registro_trafico2026-26-02.log"

frame_skip = 1
frame_id = 0

vehicle_count = 0
status = "BAJO"
color = (0,255,0)

# ---------------- KALMAN ----------------
def crear_kalman(x, y):
    kf = cv2.KalmanFilter(4,2)
    kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
    kf.transitionMatrix = np.array([[1,0,1,0],
                                    [0,1,0,1],
                                    [0,0,1,0],
                                    [0,0,0,1]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kf.statePre = np.array([[x],[y],[0],[0]], np.float32)
    return kf

# ---------------- LOG ----------------
def guardar_log(track_id, tipo, duracion):
    ahora = datetime.now()
    linea = f"{ahora},{track_id},{tipo},{duracion:.2f}s\n"
    with open(LOG_FILE, "a") as f:
        f.write(linea)

# ---------------- MOUSE ----------------
def mouse_callback(event, x, y, flags, param):
    global zona_trafico, punto_seleccionado

    if event == cv2.EVENT_LBUTTONDOWN:
        for i, p in enumerate(zona_trafico):
            if abs(p[0] - x) < 15 and abs(p[1] - y) < 15:
                punto_seleccionado = i
                return
        zona_trafico.append([x, y])

    elif event == cv2.EVENT_MOUSEMOVE:
        if punto_seleccionado is not None:
            zona_trafico[punto_seleccionado] = [x, y]

    elif event == cv2.EVENT_LBUTTONUP:
        punto_seleccionado = None

cv2.setMouseCallback("Deteccion de Vehiculos", mouse_callback)

# ---------------- CONTORNO LIGERO ----------------
def simplificar_contorno(mask):
    epsilon = 0.01 * cv2.arcLength(mask, True)
    return cv2.approxPolyDP(mask, epsilon, True)

# ---------------- DETECCIÓN ----------------
def detectar():
    global vehiculos_en_zona

    while True:
        if frame_for_detection is None:
            continue

        with lock:
            frame = frame_for_detection.copy()

        results = model.track(
            frame,
            conf=0.3,
            persist=True,
            tracker="bytetrack.yaml",
            device=device,
            imgsz=640
        )

        nuevos = {}

        if len(zona_trafico) >= 3:
            zona_np = np.array(zona_trafico, np.int32)

            for r in results:
                if r.boxes.id is None or r.masks is None:
                    continue

                ids = r.boxes.id.cpu().numpy()
                clases = r.boxes.cls.cpu().numpy()
                masks = r.masks.xy

                for mask, track_id, cls in zip(masks, ids, clases):

                    if int(cls) not in vehicle_classes:
                        continue

                    mask = np.array(mask, dtype=np.int32)

                    M = cv2.moments(mask)
                    if M["m00"] == 0:
                        continue

                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    if cv2.pointPolygonTest(zona_np, (cx, cy), False) >= 0:
                        nuevos[int(track_id)] = (mask, cls, cx, cy)

        vehiculos_en_zona = nuevos

threading.Thread(target=detectar, daemon=True).start()

# ---------------- LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.resize(frame, (1280,720))

    frame_id += 1
    if frame_id % frame_skip == 0:
        with lock:
            frame_for_detection = frame.copy()

    # Dibujar zona
    for p in zona_trafico:
        cv2.circle(frame, tuple(p), 6, (0,255,255), -1)

    if len(zona_trafico) >= 3:
        pts = np.array(zona_trafico, np.int32)
        cv2.polylines(frame, [pts], True, (0,200,255), 2)

    vehiculos_actuales = set()

    for track_id, (mask, cls, cx, cy) in vehiculos_en_zona.items():

        vehiculos_actuales.add(track_id)

        tipo = vehicle_names.get(int(cls), "Vehiculo")

        # Guardar tipo del vehículo
        if track_id not in tipo_vehiculo:
            tipo_vehiculo[track_id] = tipo

        # Kalman tracking
        if track_id not in kalman_filters:
            kalman_filters[track_id] = crear_kalman(cx, cy)

        kf = kalman_filters[track_id]
        pred = kf.predict()
        kf.correct(np.array([[np.float32(cx)], [np.float32(cy)]]))

        px, py = int(pred[0]), int(pred[1])

        mask_simple = simplificar_contorno(mask)
        cv2.polylines(frame, [mask_simple], True, (0,255,0), 2)

        cv2.circle(frame, (px, py), 5, (0,255,255), -1)

        # Historial trayectoria
        if track_id not in historial_centros:
            historial_centros[track_id] = []

        historial_centros[track_id].append((px, py))

        if len(historial_centros[track_id]) > 15:
            historial_centros[track_id].pop(0)

        for i in range(1, len(historial_centros[track_id])):
            cv2.line(frame,
                     historial_centros[track_id][i-1],
                     historial_centros[track_id][i],
                     (255,255,0), 2)

        # Entrada a zona
        if track_id not in entrada_zona:
            entrada_zona[track_id] = time.time()

        cv2.putText(frame, tipo, (px, py-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # Detectar salida
    vehiculos_salieron = vehiculos_previos - vehiculos_actuales

    for vid in vehiculos_salieron:
        if vid in entrada_zona:
            duracion = time.time() - entrada_zona[vid]

            tipo = tipo_vehiculo.get(vid, "Vehiculo")

            guardar_log(vid, tipo, duracion)

            del entrada_zona[vid]
            if vid in tipo_vehiculo:
                del tipo_vehiculo[vid]

    vehiculos_previos = vehiculos_actuales
    vehicle_count = len(vehiculos_actuales)

    # Nivel tráfico
    if vehicle_count <= 3:
        status = "BAJO"
        color = (0,255,0)
    elif vehicle_count <= 8:
        status = "MEDIO"
        color = (0,255,255)
    else:
        status = "ALTO"
        color = (0,0,255)

    cv2.putText(frame, f"Vehiculos: {vehicle_count}", (30,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.putText(frame, f"TRAFICO: {status}", (30,80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    cv2.imshow("Deteccion de Vehiculos", frame)

    key = cv2.waitKey(1)

    if key == ord('s'):
        np.save("zona_salida.npy", zona_trafico)

    if key == ord('l'):
        zona_trafico = np.load("zona_salida.npy", allow_pickle=True).tolist()

    if key == ord('c'):
        zona_trafico = []

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()