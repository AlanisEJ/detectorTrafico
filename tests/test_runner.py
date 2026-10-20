import unittest

from helpers import FakePipeline, Sink, camera, config, frame, observation
from sit.domain.types import StreamBoundary
from sit.runner import process_stream


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.camera = camera()
        self.config = config(self.camera)
        self.sink = Sink()

    def test_duplicate_frame_is_not_inferred_twice(self):
        pipeline = FakePipeline([(observation(),), (observation(),)])
        count = process_stream([frame(1, 0), frame(1, 0), frame(2, 1), StreamBoundary("eof")],
                               pipeline, self.sink, self.camera, self.config)
        self.assertEqual((count, pipeline.calls), (2, 2))
        self.assertEqual([e.event_type for e in self.sink.events], ["ENTRADA_ZONA", "SESION_CERRADA"])

    def test_reconnection_starts_new_tracker_and_session(self):
        pipeline = FakePipeline([(observation(),), (observation(),)])
        process_stream([frame(1, 0), StreamBoundary("disconnect"), frame(2, 3, epoch=2), StreamBoundary("eof")],
                       pipeline, self.sink, self.camera, self.config)
        enters = [e for e in self.sink.events if e.event_type == "ENTRADA_ZONA"]
        self.assertEqual(pipeline.resets, 1)
        self.assertNotEqual(enters[0].session_id, enters[1].session_id)
        self.assertEqual(self.sink.events[1].reason, "disconnect")
        self.assertFalse(self.sink.events[1].complete)

    def test_epoch_change_recovers_when_disconnect_message_was_dropped(self):
        pipeline = FakePipeline([(observation(),), (observation(),)])
        process_stream([frame(1, 0), frame(2, 3, epoch=2)], pipeline, self.sink, self.camera, self.config)
        self.assertEqual(pipeline.resets, 1)
        self.assertEqual(self.sink.events[1].reason, "reconnect")

    def test_max_frames_closes_without_extrapolating_an_exit(self):
        pipeline = FakePipeline([(observation(),), (observation(),)])
        self.assertEqual(process_stream([frame(1, 0), frame(2, 2)], pipeline, self.sink,
                                        self.camera, self.config, max_frames=1), 1)
        self.assertEqual(self.sink.events[-1].reason, "max_frames")
        self.assertIsNone(self.sink.events[-1].duration_s)

    def test_reader_error_is_propagated_and_closes_active_visit(self):
        pipeline = FakePipeline([(observation(),)])
        with self.assertRaises(RuntimeError):
            process_stream([frame(1, 0), StreamBoundary("error")], pipeline, self.sink, self.camera, self.config)
        self.assertEqual(self.sink.events[-1].reason, "error")

    def test_pipeline_error_is_propagated_and_closes_active_visit(self):
        pipeline = FakePipeline([(observation(),)])
        with self.assertRaises(StopIteration):
            process_stream([frame(1, 0), frame(2, 1)], pipeline, self.sink, self.camera, self.config)
        self.assertEqual(self.sink.events[-1].reason, "error")
