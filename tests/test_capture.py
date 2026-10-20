import dataclasses
import os
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from helpers import camera, frame
from sit.capture.rtsp import CaptureStream
from sit.domain.types import Frame, StreamBoundary


class FakeCap:
    def __init__(self, frames=(), opened=True):
        self.frames = iter(frames)
        self.opened = opened
        self.released = False
        self.reads = 0
        self.third_read = threading.Event()

    def isOpened(self):
        return self.opened

    def read(self):
        self.reads += 1
        if self.reads == 3:
            self.third_read.set()
        image = next(self.frames, None)
        return image is not None, image

    def set(self, *_args):
        return True

    def get(self, _key):
        return 2.0

    def release(self):
        self.released = True


class FakeCv:
    CAP_FFMPEG = 1
    CAP_PROP_OPEN_TIMEOUT_MSEC = 2
    CAP_PROP_READ_TIMEOUT_MSEC = 3
    CAP_PROP_BUFFERSIZE = 4
    CAP_PROP_FPS = 5

    def __init__(self, capture):
        self.capture = capture
        self.calls = []

    def VideoCapture(self, *args):
        self.calls.append(args)
        return self.capture

    def resize(self, image, _resolution):
        return image


class CaptureTest(unittest.TestCase):
    def test_live_queue_replaces_old_frames_without_growing(self):
        stream = CaptureStream(camera(kind="rtsp"), threading.Event(), cv2_module=FakeCv(FakeCap()))
        for seq in range(1, 4):
            stream._put(frame(seq, seq), latest=True)
        self.assertEqual(stream.messages.qsize(), 2)
        self.assertEqual([stream.messages.get_nowait().sequence for _ in range(2)], [2, 3])

    def test_video_preserves_frames_and_uses_media_time(self):
        cap = FakeCap(["one", "two", "three"])
        cam = camera(path=Path(__file__))
        with CaptureStream(cam, threading.Event(), cv2_module=FakeCv(cap)) as stream:
            messages = list(stream)
        frames = [m for m in messages if isinstance(m, Frame)]
        self.assertEqual([f.image for f in frames], ["one", "two", "three"])
        self.assertEqual([f.elapsed_s for f in frames], [0, 0.5, 1])
        self.assertEqual(messages[-1], StreamBoundary("eof"))
        self.assertTrue(cap.released)

    def test_frame_skip_does_not_change_duration_clock(self):
        cap = FakeCap(["one", "two", "three", "four", "five"])
        cam = dataclasses.replace(camera(path=Path(__file__)), frame_skip=2)
        with CaptureStream(cam, threading.Event(), cv2_module=FakeCv(cap)) as stream:
            frames = [m for m in stream if isinstance(m, Frame)]
        self.assertEqual([f.sequence for f in frames], [1, 3, 5])
        self.assertEqual([f.elapsed_s for f in frames], [0, 1, 2])

    def test_video_queue_is_bounded_and_stop_unblocks_producer(self):
        cap = FakeCap(range(1, 100))
        cam = camera(path=Path(__file__))
        with CaptureStream(cam, threading.Event(), cv2_module=FakeCv(cap)) as stream:
            self.assertTrue(cap.third_read.wait(timeout=2))
            self.assertEqual(stream.messages.qsize(), 2)
        self.assertFalse(stream.thread.is_alive())
        self.assertTrue(cap.released)

    def test_rtsp_open_failure_waits_before_reconnecting(self):
        cap = FakeCap(opened=False)
        stop = threading.Event()
        cam = camera(kind="rtsp")
        stream = CaptureStream(cam, stop, cv2_module=FakeCv(cap))
        with patch.dict(os.environ, {"SIT_TEST_RTSP": "rtsp://example.invalid/stream"}):
            with patch.object(stream, "_wait", side_effect=lambda _seconds: stop.set()) as wait:
                stream._produce()
        wait.assert_called_once_with(cam.reconnect_delay_s)
        self.assertEqual(stream.messages.get_nowait(), StreamBoundary("disconnect"))
        self.assertTrue(cap.released)

    def test_rtsp_open_uses_bounded_ffmpeg_timeouts(self):
        cv = FakeCv(FakeCap())
        cam = camera(kind="rtsp")
        stream = CaptureStream(cam, threading.Event(), cv2_module=cv)
        stream._open("rtsp://example.invalid/stream")
        self.assertEqual(cv.calls[0][1], cv.CAP_FFMPEG)
        self.assertEqual(cv.calls[0][2], [cv.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv.CAP_PROP_READ_TIMEOUT_MSEC, 5000])
