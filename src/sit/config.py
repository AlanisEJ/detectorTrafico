from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from sit.domain.zones import Zone, load_zone

SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def positive(value: object, name: str, *, integer: bool = False) -> float | int:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0
            or (integer and not isinstance(value, int))):
        raise ValueError(f"Valor inválido: {name}")
    return value


def aware_datetime(text: str) -> datetime:
    value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if value.utcoffset() is None:
        raise ValueError("start_at requiere zona horaria")
    return value


@dataclass(frozen=True)
class ModelConfig:
    weights: Path
    tracker: Path
    sha256: str | None = None
    device: str = "auto"
    half: bool = True
    confidence: float = 0.3
    image_size: int = 640
    classes: tuple[tuple[int, str], ...] = ((2, "Auto"), (3, "Moto"), (5, "Bus"), (7, "Camion"))

    def verify_weights(self) -> None:
        if not self.weights.is_file():
            raise ValueError(f"Falta el modelo local: {self.weights.name}. Consulte artifacts/models/README.md")
        if self.sha256:
            with self.weights.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest != self.sha256:
                raise ValueError("El SHA-256 del modelo no coincide")


@dataclass(frozen=True)
class CameraConfig:
    camera_id: str
    name: str
    kind: str
    source_env: str | None
    video_path: Path | None
    resolution: tuple[int, int]
    zones: tuple[Zone, ...]
    zone_paths: tuple[Path, ...]
    frame_skip: int = 1
    buffer_size: int = 2
    open_timeout_ms: int = 5000
    read_timeout_ms: int = 5000
    reconnect_delay_s: float = 2
    fps_override: float | None = None
    start_at: datetime | None = None

    def resolve_source(self) -> str:
        if self.kind == "video":
            if self.video_path is None or not self.video_path.is_file():
                raise ValueError(f"No existe el video local para {self.camera_id}")
            return str(self.video_path)
        value = os.environ.get(self.source_env or "", "")
        try:
            parsed = urlsplit(value)
            valid = parsed.scheme in ("rtsp", "rtsps") and bool(parsed.hostname)
        except ValueError:
            valid = False
        if not valid:
            raise ValueError(f"Defina una URL RTSP válida en {self.source_env}")
        return value


@dataclass(frozen=True)
class AppConfig:
    cameras: tuple[CameraConfig, ...]
    model: ModelConfig
    data_dir: Path
    timezone: ZoneInfo
    lost_timeout_s: float = 5
    exit_confirm_s: float = 0.5
    low_max: int = 3
    medium_max: int = 8


def load_config(path: Path) -> AppConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1:
        raise ValueError("Versión de configuración no soportada")
    base = path.resolve().parent
    model = raw["model"]
    sha = model.get("sha256")
    if sha is not None and not re.fullmatch(r"[a-f0-9]{64}", sha):
        raise ValueError("SHA-256 inválido")
    confidence = positive(model.get("confidence", 0.3), "confidence")
    if confidence > 1:
        raise ValueError("confidence debe ser <= 1")
    device = str(model.get("device", "auto"))
    if device not in ("auto", "cpu") and not re.fullmatch(r"\d+", device):
        raise ValueError("device debe ser auto, cpu o un índice CUDA")
    if type(model.get("half", True)) is not bool:
        raise ValueError("half debe ser booleano")
    tracker = (base / model["tracker"]).resolve()
    if not tracker.is_file():
        raise ValueError("Falta la configuración local del tracker")
    settings = ModelConfig((base / model["weights"]).resolve(), tracker, sha, device,
                           model.get("half", True), confidence,
                           positive(model.get("image_size", 640), "image_size", integer=True))
    lost = positive(raw.get("lost_timeout_s", 5), "lost_timeout_s")
    exit_confirm = raw.get("exit_confirm_s", 0.5)
    if (isinstance(exit_confirm, bool) or not isinstance(exit_confirm, (float, int))
            or not math.isfinite(exit_confirm) or exit_confirm < 0 or exit_confirm >= lost):
        raise ValueError("exit_confirm_s debe ser >= 0 y menor que lost_timeout_s")
    low, medium = raw.get("low_max", 3), raw.get("medium_max", 8)
    if type(low) is not int or type(medium) is not int or not 0 <= low < medium:
        raise ValueError("Umbrales de ocupación inválidos")
    cameras = []
    all_ids = set()
    for camera in raw["cameras"]:
        camera_id = camera["id"]
        if not isinstance(camera_id, str) or not SAFE_ID.fullmatch(camera_id) or camera_id in all_ids:
            raise ValueError("ID de cámara inválido o repetido")
        all_ids.add(camera_id)
        if type(camera.get("enabled", True)) is not bool:
            raise ValueError("enabled debe ser booleano")
        if not camera.get("enabled", True):
            continue
        source = camera["source"]
        kind = source["type"]
        if kind not in ("rtsp", "video"):
            raise ValueError("source.type debe ser rtsp o video")
        env = source.get("env")
        if kind == "rtsp" and (not isinstance(env, str) or not re.fullmatch(r"[A-Z_][A-Z0-9_]*", env)):
            raise ValueError("RTSP requiere el nombre de una variable de entorno")
        if kind == "rtsp" and any(k in source for k in ("url", "rtsp", "password", "username")):
            raise ValueError("Las credenciales RTSP deben estar fuera del JSON")
        resolution = tuple(camera["resolution"])
        if len(resolution) != 2:
            raise ValueError("resolution necesita ancho y alto")
        for size in resolution:
            positive(size, "resolution", integer=True)
        zone_paths = tuple((base / p).resolve() for p in camera["zones"])
        zones = tuple(load_zone(p) for p in zone_paths)
        if (not zones or len({z.zone_id for z in zones}) != len(zones)
                or any(z.camera_id != camera_id or z.resolution != resolution for z in zones)):
            raise ValueError("Zonas incompatibles con la cámara/resolución o repetidas")
        fps = source.get("fps")
        if fps is not None:
            positive(fps, "fps")
        cameras.append(CameraConfig(
            camera_id, str(camera.get("name", camera_id)), kind, env,
            (base / source["path"]).resolve() if kind == "video" else None,
            resolution, zones, zone_paths,
            positive(camera.get("frame_skip", 1), "frame_skip", integer=True),
            positive(camera.get("buffer_size", 2), "buffer_size", integer=True),
            positive(camera.get("open_timeout_ms", 5000), "open_timeout_ms", integer=True),
            positive(camera.get("read_timeout_ms", 5000), "read_timeout_ms", integer=True),
            positive(camera.get("reconnect_delay_s", 2), "reconnect_delay_s"), fps,
            aware_datetime(source["start_at"]) if "start_at" in source else None))
    if not cameras:
        raise ValueError("Se requiere al menos una cámara habilitada")
    return AppConfig(tuple(cameras), settings, (base / raw.get("data_dir", "../data")).resolve(),
                     ZoneInfo(raw.get("timezone", "UTC")), lost, exit_confirm, low, medium)
