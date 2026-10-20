from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from sit.domain.types import Point


def cross(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def on_segment(p: Point, a: Point, b: Point) -> bool:
    return (abs(cross(a, b, p)) < 1e-7
            and min(a[0], b[0]) - 1e-7 <= p[0] <= max(a[0], b[0]) + 1e-7
            and min(a[1], b[1]) - 1e-7 <= p[1] <= max(a[1], b[1]) + 1e-7)


def segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    return ((cross(a, b, c) * cross(a, b, d) < 0
             and cross(c, d, a) * cross(c, d, b) < 0)
            or any((on_segment(c, a, b), on_segment(d, a, b),
                    on_segment(a, c, d), on_segment(b, c, d))))


def centroid(points: tuple[Point, ...]) -> Point:
    if len(points) < 3 or any(not math.isfinite(v) for point in points for v in point):
        raise ValueError("Contorno inválido")
    pairs = tuple(zip(points, points[1:] + points[:1]))
    twice_area = sum(a[0] * b[1] - b[0] * a[1] for a, b in pairs)
    if abs(twice_area) < 1e-7:
        raise ValueError("Polígono sin área")
    return tuple(sum((a[i] + b[i]) * (a[0] * b[1] - b[0] * a[1])
                     for a, b in pairs) / (3 * twice_area) for i in (0, 1))


@dataclass(frozen=True)
class Zone:
    zone_id: str
    camera_id: str
    resolution: tuple[int, int]
    points: tuple[Point, ...]

    def __post_init__(self) -> None:
        if not self.zone_id or not self.camera_id:
            raise ValueError("La zona necesita id y camera_id")
        if len(self.resolution) != 2 or any(type(x) is not int or x <= 0 for x in self.resolution):
            raise ValueError("Resolución de zona inválida")
        if len(self.points) < 3 or len(set(self.points)) != len(self.points):
            raise ValueError("Se requieren al menos tres puntos distintos, sin repeticiones")
        width, height = self.resolution
        for point in self.points:
            if len(point) != 2 or any(not math.isfinite(v) for v in point):
                raise ValueError("Coordenadas de zona inválidas")
            if not (0 <= point[0] < width and 0 <= point[1] < height):
                raise ValueError("Punto fuera de la resolución de referencia")
        edges = tuple(zip(self.points, self.points[1:] + self.points[:1]))
        for i, (a, b) in enumerate(edges):
            for j, (c, d) in enumerate(edges):
                if j <= i + 1 or (i == 0 and j == len(edges) - 1):
                    continue
                if segments_intersect(a, b, c, d):
                    raise ValueError("La zona se cruza consigo misma")
        centroid(self.points)

    def contains(self, point: Point) -> bool:
        inside = False
        for a, b in zip(self.points, self.points[1:] + self.points[:1]):
            if on_segment(point, a, b):
                return True
            if (a[1] > point[1]) != (b[1] > point[1]):
                x = a[0] + (point[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
                if point[0] < x:
                    inside = not inside
        return inside


def load_zone(path: Path) -> Zone:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError(f"Versión de zona no soportada: {path.name}")
    return Zone(str(data["id"]), str(data["camera_id"]),
                tuple(data["resolution"]), tuple(tuple(p) for p in data["points"]))


def save_zone(path: Path, zone: Zone) -> None:
    payload = {"schema_version": 1, "id": zone.zone_id, "camera_id": zone.camera_id,
               "resolution": zone.resolution, "points": zone.points}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
