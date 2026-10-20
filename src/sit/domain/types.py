from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

Point = tuple[float, float]


@dataclass(frozen=True)
class Observation:
    track_id: int
    class_id: int
    label: str
    center: Point
    contour: tuple[Point, ...] = ()


@dataclass(frozen=True)
class Frame:
    camera_id: str
    sequence: int
    elapsed_s: float
    timestamp: datetime
    epoch: int = 1
    image: Any = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class StreamBoundary:
    reason: str


@dataclass(frozen=True)
class TrafficEvent:
    schema_version: int
    event_id: str
    session_id: str
    visit_id: str
    camera_id: str
    zone_id: str
    track_id: int
    class_id: int
    vehicle_type: str
    event_type: str
    timestamp_utc: datetime
    frame_id: int
    duration_s: float | None
    observed_duration_s: float
    complete: bool
    reason: str
