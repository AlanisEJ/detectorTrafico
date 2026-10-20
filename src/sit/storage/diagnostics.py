from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


class RedactRtsp(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = re.sub(r"rtsps?://[^\s\"']+", "<RTSP omitido>", record.getMessage(), flags=re.I)
        record.args = ()
        return True


class DailyLog(logging.Handler):
    def __init__(self, root: Path, timezone: ZoneInfo):
        super().__init__()
        self.root, self.timezone = root, timezone

    def emit(self, record: logging.LogRecord) -> None:
        date = datetime.fromtimestamp(record.created, self.timezone).date().isoformat()
        path = self.root / date / "logs" / "trafico.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(self.format(record) + "\n")


def configure_logging(root: Path, timezone: ZoneInfo) -> logging.Logger:
    logger = logging.getLogger("sit")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    for handler in (logging.StreamHandler(), DailyLog(root, timezone)):
        handler.addFilter(RedactRtsp())
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    return logger
