from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import timezone
from uuid import NAMESPACE_URL, uuid4, uuid5

from sit.domain.types import Frame, Observation, TrafficEvent
from sit.domain.zones import Zone


@dataclass
class Visit:
    visit_id: str
    observation: Observation
    entered: Frame
    last_seen: Frame
    last_inside: Frame
    outside_since: Frame | None = None


class ZoneEvents:
    """Una instancia por cámara/sesión. Solamente consume frames nuevos."""

    def __init__(self, camera_id: str, zones: tuple[Zone, ...], *,
                 session_id: str | None = None, lost_timeout_s: float = 5,
                 exit_confirm_s: float = 0.5):
        if (not math.isfinite(lost_timeout_s) or lost_timeout_s <= 0
                or not math.isfinite(exit_confirm_s) or exit_confirm_s < 0):
            raise ValueError("Tiempos de seguimiento inválidos")
        if any(z.camera_id != camera_id for z in zones):
            raise ValueError("Zona de otra cámara")
        if len({z.zone_id for z in zones}) != len(zones):
            raise ValueError("IDs de zona repetidos")
        self.camera_id = camera_id
        self.zones = zones
        self.session_id = session_id or str(uuid4())
        self.lost_timeout_s = lost_timeout_s
        self.exit_confirm_s = exit_confirm_s
        self.active: dict[tuple[str, int], Visit] = {}
        self.last_frame: Frame | None = None

    def _event(self, zone_id: str, visit: Visit, kind: str, frame: Frame,
               reason: str, complete: bool = False) -> TrafficEvent:
        observed = max(0.0, visit.last_inside.elapsed_s - visit.entered.elapsed_s)
        duration = max(0.0, frame.elapsed_s - visit.entered.elapsed_s) if complete else None
        if kind == "ENTRADA_ZONA":
            duration = 0.0
        identity = f"sit:{self.session_id}:{visit.visit_id}:{kind}"
        return TrafficEvent(1, str(uuid5(NAMESPACE_URL, identity)), self.session_id,
                            visit.visit_id, self.camera_id, zone_id,
                            visit.observation.track_id, visit.observation.class_id,
                            visit.observation.label, kind, frame.timestamp.astimezone(timezone.utc),
                            frame.sequence, duration, observed, complete, reason)

    def process(self, frame: Frame, observations: tuple[Observation, ...]) -> list[TrafficEvent]:
        # Los estados temporales conservan metadatos, nunca imágenes por vehículo.
        frame = replace(frame, image=None)
        if frame.camera_id != self.camera_id:
            raise ValueError("Frame de otra cámara")
        if not math.isfinite(frame.elapsed_s) or frame.elapsed_s < 0:
            raise ValueError("Tiempo de frame inválido")
        if frame.timestamp.utcoffset() is None:
            raise ValueError("El timestamp requiere zona horaria")
        if self.last_frame:
            if frame.sequence <= self.last_frame.sequence:
                return []
            if frame.epoch != self.last_frame.epoch:
                raise ValueError("Una reconexión requiere una sesión nueva")
            if frame.elapsed_s < self.last_frame.elapsed_s:
                raise ValueError("El tiempo de frames debe ser monótono")
        if len({o.track_id for o in observations}) != len(observations):
            raise ValueError("IDs repetidos en el mismo frame")
        self.last_frame = frame
        events = []
        for key, visit in tuple(self.active.items()):
            if frame.elapsed_s - visit.last_seen.elapsed_s > self.lost_timeout_s:
                events.append(self._event(key[0], visit, "SEGUIMIENTO_PERDIDO", frame,
                                          "lost_timeout"))
                del self.active[key]

        for zone in self.zones:
            for observation in observations:
                key = (zone.zone_id, observation.track_id)
                visit = self.active.get(key)
                if zone.contains(observation.center):
                    if visit is None:
                        visit = Visit(str(uuid4()), observation, frame, frame, frame)
                        self.active[key] = visit
                        events.append(self._event(zone.zone_id, visit, "ENTRADA_ZONA", frame,
                                                  "observed_inside"))
                    visit.last_seen = visit.last_inside = frame
                    visit.outside_since = None
                elif visit is not None:
                    visit.last_seen = frame
                    if visit.outside_since is None:
                        visit.outside_since = frame
                    if frame.elapsed_s - visit.outside_since.elapsed_s >= self.exit_confirm_s:
                        events.append(self._event(zone.zone_id, visit, "ZONA_TRAFICO",
                                                  visit.outside_since, "confirmed_outside", True))
                        del self.active[key]
        return events

    def close(self, reason: str) -> list[TrafficEvent]:
        if self.last_frame is None:
            return []
        events = [self._event(zone_id, visit, "SESION_CERRADA", self.last_frame, reason)
                  for (zone_id, _), visit in self.active.items()]
        self.active.clear()
        return events


def traffic_level(count: int, low_max: int = 3, medium_max: int = 8) -> str:
    return "BAJO" if count <= low_max else "MEDIO" if count <= medium_max else "ALTO"
