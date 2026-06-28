import queue
import threading
import time
from enum import Enum
from typing import Callable, Optional

import cv2

SourceCallback = Callable[[str], None]
PlaybackCallback = Callable[[bool], None]
PositionCallback = Callable[[float, float], None]
ErrorCallback = Callable[[str], None]


class SourceMode(Enum):
    FILE = "file"
    WEBCAM = "webcam"
    CONTINUITY = "continuity"


SPEED_OPTIONS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


def queue_put_drop_oldest(q: queue.Queue, item) -> None:
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass


class SourceManager:
    def __init__(
        self,
        panel_queues: list[queue.Queue],
        on_source_changed: Optional[SourceCallback] = None,
        on_playback_changed: Optional[PlaybackCallback] = None,
        on_position_changed: Optional[PositionCallback] = None,
        on_error: Optional[ErrorCallback] = None,
    ):
        self._panel_queues = panel_queues
        self._on_source_changed = on_source_changed
        self._on_playback_changed = on_playback_changed
        self._on_position_changed = on_position_changed
        self._on_error = on_error
        self._cap: Optional[cv2.VideoCapture] = None
        self._mode = SourceMode.WEBCAM
        self._source_label = "Webcam 0"
        self._file_path: Optional[str] = None
        self._camera_index = 0
        self._playing = False
        self._loop = False
        self._speed_index = 3
        self._seek_request: Optional[float] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._frame_number = 0
        self._duration_sec = 0.0
        self._fps = 30.0
        self._initializing = False
        self._ui_emit_counter = 0

    @property
    def mode(self) -> SourceMode:
        return self._mode

    @property
    def is_playing(self) -> bool:
        return self._playing

    @property
    def is_looping(self) -> bool:
        return self._loop

    @property
    def speed(self) -> float:
        return SPEED_OPTIONS[self._speed_index]

    @property
    def duration_sec(self) -> float:
        return self._duration_sec

    @property
    def source_label(self) -> str:
        return self._source_label

    def set_speed_index(self, index: int) -> None:
        self._speed_index = max(0, min(index, len(SPEED_OPTIONS) - 1))

    def set_loop(self, enabled: bool) -> None:
        self._loop = enabled

    def toggle_play(self) -> None:
        if self._cap is None or not self._cap.isOpened():
            return
        self._playing = not self._playing
        if self._on_playback_changed:
            self._on_playback_changed(self._playing)

    def pause(self) -> None:
        self._playing = False
        if self._on_playback_changed:
            self._on_playback_changed(False)

    def play(self) -> None:
        if self._cap is None or not self._cap.isOpened():
            return
        self._playing = True
        if self._on_playback_changed:
            self._on_playback_changed(True)

    def seek(self, position_sec: float) -> None:
        if self._mode != SourceMode.FILE:
            return
        self._seek_request = max(0.0, min(position_sec, self._duration_sec))

    def _stop_thread(self) -> None:
        self._running = False
        self._release_capture()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def _clear_queues(self) -> None:
        for q in self._panel_queues:
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

    def _release_capture(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _open_capture(self) -> bool:
        self._release_capture()
        self._frame_number = 0

        if self._mode == SourceMode.FILE:
            if not self._file_path:
                return False
            cap = cv2.VideoCapture(self._file_path, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                cap = cv2.VideoCapture(self._file_path)
            if not cap.isOpened():
                return False
            self._cap = cap
            self._fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            if self._fps <= 0 or self._fps > 240:
                self._fps = 30.0
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            self._duration_sec = frame_count / self._fps if frame_count > 0 else 0.0
            self._source_label = self._file_path.rsplit("/", 1)[-1]
        else:
            cap = cv2.VideoCapture(self._camera_index)
            if not cap.isOpened():
                return False
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._cap = cap
            self._fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            if self._fps <= 0 or self._fps > 120:
                self._fps = 30.0
            self._duration_sec = 0.0
            if self._mode == SourceMode.WEBCAM:
                self._source_label = f"Webcam {self._camera_index}"
            else:
                self._source_label = f"Camera {self._camera_index}"

        if self._on_source_changed:
            self._on_source_changed(self._source_label)
        return True

    def switch_to_file(self, file_path: str, resume: bool = False) -> None:
        was_playing = self._playing or resume
        self._initializing = True
        self.pause()
        self._stop_thread()
        self._clear_queues()
        self._mode = SourceMode.FILE
        self._file_path = file_path
        if not self._open_capture():
            self._initializing = False
            if self._on_error:
                self._on_error(f"Failed to open file: {file_path.rsplit('/', 1)[-1]}")
            return
        self._initializing = False
        self._start_thread()
        if was_playing:
            self.play()

    def switch_to_webcam(self, camera_index: int = 0) -> None:
        self._initializing = True
        self.pause()
        self._stop_thread()
        self._clear_queues()
        self._mode = SourceMode.WEBCAM
        self._camera_index = camera_index
        self._file_path = None
        if not self._open_capture():
            self._initializing = False
            if self._on_error:
                self._on_error(f"Failed to open webcam {camera_index}")
            return
        self._initializing = False
        self._start_thread()
        self.play()

    def switch_to_continuity(self, camera_index: int) -> None:
        self._initializing = True
        self.pause()
        self._stop_thread()
        self._clear_queues()
        self._mode = SourceMode.CONTINUITY
        self._camera_index = camera_index
        self._file_path = None
        if not self._open_capture():
            self._initializing = False
            if self._on_error:
                self._on_error(f"Failed to open camera {camera_index}")
            return
        self._initializing = False
        self._start_thread()
        self.play()

    def _start_thread(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _reader_loop(self) -> None:
        while self._running:
            if self._initializing:
                time.sleep(0.05)
                continue

            with self._lock:
                cap = self._cap
                mode = self._mode
                playing = self._playing
                if cap is not None and self._seek_request is not None:
                    cap.set(cv2.CAP_PROP_POS_MSEC, self._seek_request * 1000.0)
                    self._seek_request = None

            if cap is None or not cap.isOpened():
                time.sleep(0.05)
                continue

            if not playing:
                time.sleep(0.005)
                continue

            if mode == SourceMode.FILE:
                ret, frame = cap.read()
            else:
                for _ in range(2):
                    if not cap.grab():
                        break
                ret, frame = cap.retrieve()
                if not ret or frame is None:
                    ret, frame = cap.read()

            if not self._running:
                break

            if not ret or frame is None:
                if mode == SourceMode.FILE:
                    if self._loop and self._cap is not None:
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self._frame_number = 0
                        continue
                    self._playing = False
                    if self._on_playback_changed:
                        self._on_playback_changed(False)
                time.sleep(0.02)
                continue

            self._frame_number += 1
            timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            h, w = frame.shape[:2]

            metadata = {
                "source_label": self._source_label,
                "frame_number": self._frame_number,
                "resolution": f"{w}x{h}",
                "timestamp_sec": timestamp_sec,
                "total_sec": self._duration_sec,
                "mode": mode.value,
            }

            packet = (frame, metadata)
            for q in self._panel_queues:
                queue_put_drop_oldest(q, packet)

            self._ui_emit_counter += 1
            if self._on_position_changed and (
                self._ui_emit_counter % 3 == 0 or mode == SourceMode.FILE
            ):
                self._on_position_changed(timestamp_sec, self._duration_sec)

            if mode == SourceMode.FILE:
                delay = (1.0 / self._fps) / self.speed
                time.sleep(delay)

    def stop(self) -> None:
        self._running = False
        self._playing = False
        self._release_capture()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
