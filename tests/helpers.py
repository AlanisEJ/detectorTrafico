from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from sit.config import AppConfig, CameraConfig, ModelConfig
from sit.domain.types import Frame, Observation
from sit.domain.zones import Zone

ROOT = Path(__file__).resolve().parents[1]
START = datetime(2026, 2, 26, 15, tzinfo=timezone.utc)


def zone(camera="cam_1"):
    return Zone("zona", camera, (160, 120), ((0, 0), (100, 0), (100, 100), (0, 100)))


def camera(kind="video", path=None, camera_id="cam_1"):
    return CameraConfig(camera_id, "Prueba", kind, "SIT_TEST_RTSP" if kind == "rtsp" else None,
                        path or ROOT / "README.md", (160, 120), (zone(camera_id),), (Path("zone.json"),),
                        reconnect_delay_s=0.01)


def config(cam=None):
    cam = cam or camera()
    return AppConfig((cam,), ModelConfig(Path("missing-seg.pt"), Path("tracker.yaml")),
                     Path("unused"), ZoneInfo("UTC"))


def frame(seq, elapsed, camera_id="cam_1", epoch=1, image=None):
    return Frame(camera_id, seq, elapsed, START + timedelta(seconds=elapsed), epoch, image)


def observation(track_id=1, point=(50, 50), label="Auto"):
    return Observation(track_id, 2, label, point)


class Sink:
    def __init__(self):
        self.events = []

    def write(self, events):
        self.events.extend(events)


class FakePipeline:
    def __init__(self, observations):
        self.observations = iter(observations)
        self.calls = 0
        self.resets = 0

    def infer(self, _image):
        self.calls += 1
        return next(self.observations)

    def reset(self):
        self.resets += 1
