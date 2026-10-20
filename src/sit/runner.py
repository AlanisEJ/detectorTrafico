from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from sit.config import AppConfig, CameraConfig
from sit.domain.events import ZoneEvents, traffic_level
from sit.domain.types import Frame, Observation, StreamBoundary, TrafficEvent


class Pipeline(Protocol):
    def infer(self, image: object) -> tuple[Observation, ...]: ...
    def reset(self) -> None: ...


class EventSink(Protocol):
    def write(self, events: list[TrafficEvent]) -> None: ...


def process_stream(messages: Iterable[Frame | StreamBoundary], pipeline: Pipeline,
                   sink: EventSink, camera: CameraConfig, config: AppConfig,
                   *, max_frames: int | None = None, logger=None) -> int:
    engine = None
    epoch = None
    seen_before = False
    processed = 0
    reason = "shutdown"
    last_sequence = -1
    last_report = float("-inf")
    try:
        for message in messages:
            if isinstance(message, StreamBoundary):
                if logger and message.reason == "disconnect":
                    logger.warning("Cámara %s desconectada; esperando reconexión", camera.camera_id)
                if engine:
                    sink.write(engine.close(message.reason))
                    engine = None
                if message.reason == "error":
                    raise RuntimeError("Falló el lector de video; revise conectividad/backend")
                if message.reason == "eof":
                    reason = "eof"
                    break
                continue
            if message.sequence <= last_sequence:
                continue
            last_sequence = message.sequence
            if engine is None or message.epoch != epoch:
                if engine:
                    sink.write(engine.close("reconnect"))
                if seen_before:
                    pipeline.reset()
                engine = ZoneEvents(camera.camera_id, camera.zones,
                                    lost_timeout_s=config.lost_timeout_s,
                                    exit_confirm_s=config.exit_confirm_s)
                epoch = message.epoch
                seen_before = True
            observations = pipeline.infer(message.image)
            sink.write(engine.process(message, observations))
            if logger and message.elapsed_s - last_report >= 10:
                occupancy = sum(any(z.contains(o.center) for z in camera.zones) for o in observations)
                logger.info("Cámara %s ocupacion=%s nivel=%s frame=%s", camera.camera_id,
                            occupancy, traffic_level(occupancy, config.low_max, config.medium_max),
                            message.sequence)
                last_report = message.elapsed_s
            processed += 1
            if max_frames is not None and processed >= max_frames:
                reason = "max_frames"
                break
    except Exception:
        reason = "error"
        raise
    finally:
        if engine:
            sink.write(engine.close(reason))
    return processed
