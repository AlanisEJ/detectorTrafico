import csv
import dataclasses
import logging
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from helpers import frame, observation, zone
from sit.domain.events import ZoneEvents
from sit.storage.csv_events import CsvEvents, EVENT_COLUMNS
from sit.storage.diagnostics import RedactRtsp


class StorageTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.sink = CsvEvents(Path(self.tmp.name), ZoneInfo("America/Hermosillo"))
        engine = ZoneEvents("cam_1", (zone(),))
        self.entered = engine.process(frame(1, 0), (observation(),))[0]
        self.closed = engine.close("shutdown")[0]

    def test_csv_schema_numeric_duration_and_incomplete_value(self):
        self.sink.write([self.entered, self.closed])
        with self.sink.path_for(self.entered).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            self.assertEqual(tuple(reader.fieldnames), EVENT_COLUMNS)
        self.assertEqual(rows[0]["duration_s"], "0.000000")
        self.assertEqual(rows[1]["duration_s"], "")
        self.assertEqual(rows[1]["complete"], "false")
        self.assertTrue(rows[0]["timestamp_utc"].endswith("+00:00"))

    def test_daily_partition_uses_configured_timezone(self):
        event = dataclasses.replace(self.entered, timestamp_utc=datetime(2026, 2, 27, 2, tzinfo=timezone.utc))
        self.assertIn("2026-02-26", self.sink.path_for(event).parts)

    def test_rejects_existing_mixed_schema(self):
        path = self.sink.path_for(self.entered)
        path.parent.mkdir(parents=True)
        path.write_text("old,format\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.sink.write([self.entered])
        self.assertEqual(path.read_text(), "old,format\n")

    def test_writer_keeps_one_header_under_concurrent_cameras(self):
        threads = [threading.Thread(target=self.sink.write, args=([self.entered],)) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        with self.sink.path_for(self.entered).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        self.assertEqual(len(rows), 9)
        self.assertEqual(rows.count(list(EVENT_COLUMNS)), 1)

    def test_rejects_path_traversal_in_camera_id(self):
        with self.assertRaises(ValueError):
            self.sink.write([dataclasses.replace(self.entered, camera_id="../escape")])

    def test_diagnostic_filter_removes_rtsp_credentials_and_host(self):
        record = logging.LogRecord("sit", logging.ERROR, __file__, 0,
                                   "Falló %s", ("rtsp://sample_user:sample_password@127.0.0.1/video",), None)
        self.assertTrue(RedactRtsp().filter(record))
        self.assertNotIn("sample_password", record.getMessage())
        self.assertNotIn("127.0.0.1", record.getMessage())
