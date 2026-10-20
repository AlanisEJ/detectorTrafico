from __future__ import annotations

import csv
import threading
from dataclasses import asdict, fields
from pathlib import Path
from zoneinfo import ZoneInfo

from sit.config import SAFE_ID
from sit.domain.types import TrafficEvent

EVENT_COLUMNS = tuple(field.name for field in fields(TrafficEvent))


class CsvEvents:
    def __init__(self, root: Path, timezone: ZoneInfo):
        self.root, self.timezone = root, timezone
        self.lock = threading.Lock()

    def path_for(self, event: TrafficEvent) -> Path:
        if not SAFE_ID.fullmatch(event.camera_id):
            raise ValueError("camera_id inválido para almacenamiento")
        date = event.timestamp_utc.astimezone(self.timezone).date().isoformat()
        return self.root / date / "database" / event.camera_id / "trafico.csv"

    def write(self, events: list[TrafficEvent]) -> None:
        with self.lock:
            for event in events:
                path = self.path_for(event)
                path.parent.mkdir(parents=True, exist_ok=True)
                exists = path.exists() and path.stat().st_size > 0
                if exists:
                    with path.open(newline="", encoding="utf-8") as handle:
                        if tuple(next(csv.reader(handle))) != EVENT_COLUMNS:
                            raise ValueError("CSV existente tiene otro esquema; no se mezclaron datos")
                row = asdict(event)
                row["timestamp_utc"] = event.timestamp_utc.isoformat(timespec="milliseconds")
                for name in ("duration_s", "observed_duration_s"):
                    row[name] = "" if row[name] is None else f"{row[name]:.6f}"
                row["complete"] = str(row["complete"]).lower()
                with path.open("a", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=EVENT_COLUMNS)
                    if not exists:
                        writer.writeheader()
                    writer.writerow(row)
