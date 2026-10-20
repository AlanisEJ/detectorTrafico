import dataclasses
import unittest

from helpers import frame, observation, zone
from sit.domain.events import ZoneEvents, traffic_level


class EventsTest(unittest.TestCase):
    def setUp(self):
        self.engine = ZoneEvents("cam_1", (zone(),), session_id="test")

    def enter(self):
        return self.engine.process(frame(1, 0), (observation(),))

    def test_confirmed_exit_uses_first_outside_time(self):
        entered = self.enter()[0]
        self.assertEqual(entered.event_type, "ENTRADA_ZONA")
        self.assertEqual(self.engine.process(frame(2, 2), (observation(point=(120, 50)),)), [])
        event, = self.engine.process(frame(3, 2.6), (observation(point=(125, 50)),))
        self.assertEqual((event.event_type, event.duration_s, event.complete), ("ZONA_TRAFICO", 2, True))
        self.assertEqual(event.frame_id, 2)
        self.assertEqual(event.visit_id, entered.visit_id)
        self.assertNotEqual(event.event_id, entered.event_id)
        self.assertFalse(self.engine.active)

    def test_temporary_missing_track_does_not_split_visit(self):
        self.enter()
        self.assertEqual(self.engine.process(frame(2, 1), ()), [])
        self.assertEqual(self.engine.process(frame(3, 3), (observation(),)), [])
        self.assertEqual(len(self.engine.active), 1)

    def test_long_missing_track_is_incomplete(self):
        self.enter()
        event, = self.engine.process(frame(2, 5.1), ())
        self.assertEqual(event.event_type, "SEGUIMIENTO_PERDIDO")
        self.assertIsNone(event.duration_s)
        self.assertFalse(event.complete)
        self.assertFalse(self.engine.active)

    def test_reappearance_after_timeout_starts_new_visit(self):
        first = self.enter()[0]
        lost, entered = self.engine.process(frame(2, 6), (observation(),))
        self.assertEqual(lost.event_type, "SEGUIMIENTO_PERDIDO")
        self.assertEqual(entered.event_type, "ENTRADA_ZONA")
        self.assertNotEqual(first.visit_id, entered.visit_id)

    def test_short_boundary_jitter_does_not_emit_exit(self):
        self.enter()
        self.engine.process(frame(2, 1), (observation(point=(101, 50)),))
        self.assertEqual(self.engine.process(frame(3, 1.2), (observation(point=(99, 50)),)), [])
        self.assertEqual(len(self.engine.active), 1)

    def test_reentry_after_confirmed_exit_is_new_visit(self):
        first = self.enter()[0]
        self.engine.process(frame(2, 1), (observation(point=(110, 50)),))
        self.engine.process(frame(3, 2), (observation(point=(110, 50)),))
        second, = self.engine.process(frame(4, 3), (observation(),))
        self.assertNotEqual(first.visit_id, second.visit_id)

    def test_duplicate_or_old_frames_do_not_advance_state(self):
        self.enter()
        self.assertEqual(self.engine.process(frame(1, 1), ()), [])
        self.assertEqual(self.engine.process(frame(0, 2), ()), [])
        self.assertEqual(self.engine.last_frame.elapsed_s, 0)

    def test_rejects_nonmonotonic_time_and_new_epoch(self):
        self.enter()
        self.engine.process(frame(2, 1), (observation(),))
        with self.assertRaises(ValueError):
            self.engine.process(frame(3, 0.5), ())
        with self.assertRaises(ValueError):
            self.engine.process(frame(3, 2, epoch=2), ())

    def test_rejects_wrong_camera_naive_time_and_duplicate_ids(self):
        with self.assertRaises(ValueError):
            self.engine.process(frame(1, 0, camera_id="cam_2"), ())
        packet = frame(1, 0)
        with self.assertRaises(ValueError):
            self.engine.process(dataclasses.replace(packet, timestamp=packet.timestamp.replace(tzinfo=None)), ())
        with self.assertRaises(ValueError):
            self.engine.process(packet, (observation(), observation()))

    def test_shutdown_does_not_claim_an_exit_and_is_idempotent(self):
        self.enter()
        self.engine.process(frame(2, 2), (observation(),))
        event, = self.engine.close("shutdown")
        self.assertEqual((event.event_type, event.reason), ("SESION_CERRADA", "shutdown"))
        self.assertIsNone(event.duration_s)
        self.assertEqual(event.observed_duration_s, 2)
        self.assertEqual(self.engine.close("shutdown"), [])

    def test_track_ids_are_isolated_by_camera(self):
        self.enter()
        other = ZoneEvents("cam_2", (zone("cam_2"),))
        other.process(frame(1, 0, camera_id="cam_2"), (observation(),))
        self.engine.close("disconnect")
        self.assertEqual(len(other.active), 1)

    def test_does_not_keep_images_in_tracking_state(self):
        image = object()
        self.engine.process(frame(1, 0, image=image), (observation(),))
        self.assertIsNone(self.engine.last_frame.image)
        self.assertIsNone(next(iter(self.engine.active.values())).entered.image)

    def test_missing_tracks_are_cleaned_from_memory(self):
        observations = tuple(observation(track_id=i) for i in range(100))
        self.engine.process(frame(1, 0), observations)
        self.engine.process(frame(2, 6), ())
        self.assertEqual(len(self.engine.active), 0)

    def test_occupancy_thresholds_match_prototype(self):
        self.assertEqual([traffic_level(i) for i in (0, 3, 4, 8, 9)], ["BAJO", "BAJO", "MEDIO", "MEDIO", "ALTO"])
