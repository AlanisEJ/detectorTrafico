from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from sit.domain.events import ZoneEvents
from sit.domain.types import Frame, Observation
from sit.domain.zones import Zone
from sit.storage.csv_events import CsvEvents


def run_demo(output: Path) -> tuple[int, Path]:
    zone = Zone("zona_demo", "demo", (160, 120), ((0, 0), (100, 0), (100, 100), (0, 100)))
    engine = ZoneEvents("demo", (zone,))
    sink = CsvEvents(output, ZoneInfo("UTC"))
    start = datetime.now(timezone.utc)
    total, last_event = 0, None
    # Pérdida breve recuperada, salida confirmada, pérdida prolongada y cierre incompleto.
    script = [(0, 1, 50), (1, 1, 50), (2, None, 0), (3, 1, 50),
              (4, 1, 120), (4.6, 1, 125), (5, 2, 50), (6, None, 0),
              (10.1, None, 0), (12, 3, 50)]
    for sequence, (elapsed, track_id, x) in enumerate(script, 1):
        observations = () if track_id is None else (Observation(track_id, 2, "Auto", (x, 50)),)
        frame = Frame("demo", sequence, elapsed, start + timedelta(seconds=elapsed))
        events = engine.process(frame, observations)
        sink.write(events)
        total += len(events)
        if events:
            last_event = events[-1]
    events = engine.close("eof")
    sink.write(events)
    total += len(events)
    return total, sink.path_for(events[-1] if events else last_event)
