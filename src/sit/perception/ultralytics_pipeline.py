from __future__ import annotations

import os
from typing import Any

from sit.config import ModelConfig
from sit.domain.types import Observation
from sit.domain.zones import centroid


class UltralyticsPipeline:
    """Un modelo y tracker por cámara; la instancia tiene un único consumidor."""

    def __init__(self, settings: ModelConfig, *, model_factory=None, torch_module=None):
        settings.verify_weights()
        os.environ["YOLO_AUTOINSTALL"] = "false"
        if model_factory is None:
            from ultralytics import YOLO
            model_factory = YOLO
        if torch_module is None:
            import torch
            torch_module = torch
        self.settings = settings
        self.device = ("0" if torch_module.cuda.is_available() else "cpu") if settings.device == "auto" else settings.device
        self.half = settings.half and self.device != "cpu"
        self.factory = model_factory
        self.model = self.factory(str(settings.weights))
        self._check_task()

    def _check_task(self) -> None:
        if self.model.task != "segment":
            raise ValueError("Se requiere un modelo de segmentación, no uno de detección por cajas")

    def reset(self) -> None:
        # API pública: reemplazar el modelo también elimina el estado persistente de ByteTrack.
        self.model = self.factory(str(self.settings.weights))
        self._check_task()

    def infer(self, image: Any) -> tuple[Observation, ...]:
        labels = dict(self.settings.classes)
        results = self.model.track(
            image, conf=self.settings.confidence, persist=True,
            tracker=str(self.settings.tracker), device=self.device,
            half=self.half, imgsz=self.settings.image_size,
            classes=list(labels), verbose=False, save=False, show=False)
        observations = []
        for result in results:
            if result.boxes is None or result.boxes.id is None:
                continue
            ids = result.boxes.id.cpu().tolist()
            classes = result.boxes.cls.cpu().tolist()
            if result.masks is None:
                if ids:
                    raise ValueError("El modelo devolvió tracks sin máscaras")
                continue
            for mask, track_id, class_id in zip(result.masks.xy, ids, classes, strict=True):
                if int(class_id) not in labels:
                    continue
                contour = tuple((float(point[0]), float(point[1])) for point in mask)
                try:
                    center = centroid(contour)
                except ValueError:
                    continue
                observations.append(Observation(int(track_id), int(class_id), labels[int(class_id)], center, contour))
        return tuple(observations)
