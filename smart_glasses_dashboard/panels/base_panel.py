import queue
import threading
from abc import abstractmethod
from typing import Callable, Optional

import cv2
import numpy as np

from utils.fps_counter import FpsCounter
from utils.frame_utils import draw_header_bar_inplace, frame_to_jpeg_bytes, make_placeholder

FrameCallback = Callable[[bytes, dict, list], None]


class PanelWorker(threading.Thread):
    def __init__(
        self,
        panel_name: str,
        accent_color: tuple[int, int, int],
        input_queue: queue.Queue,
        on_frame: Optional[FrameCallback] = None,
        panel_width: int = 640,
        panel_height: int = 480,
    ):
        super().__init__(daemon=True)
        self.panel_name = panel_name
        self.accent_color = accent_color
        self.input_queue = input_queue
        self.on_frame = on_frame
        self.panel_width = panel_width
        self.panel_height = panel_height
        self._enabled = True
        self._running = True
        self._fps_counter = FpsCounter()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        placeholder = make_placeholder(
            self.panel_width,
            self.panel_height,
            self.panel_name,
            self.accent_color,
        )
        self._emit(placeholder, {}, [])

        while self._running:
            try:
                frame, metadata = self.input_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            while True:
                try:
                    frame, metadata = self.input_queue.get_nowait()
                except queue.Empty:
                    break

            try:
                if self._enabled:
                    processed, stats = self.process(frame, metadata)
                else:
                    processed, stats = self.process_disabled(frame, metadata)
            except Exception:
                processed = frame.copy()
                stats = {}

            self._fps_counter.tick()
            header_lines = self.build_header_lines(stats, metadata)
            if processed is not frame:
                display = draw_header_bar_inplace(
                    processed,
                    self.panel_name,
                    self.accent_color,
                    header_lines,
                )
            else:
                display = draw_header_bar_inplace(
                    frame.copy(),
                    self.panel_name,
                    self.accent_color,
                    header_lines,
                )
            self._emit(display, stats, header_lines)

    def _emit(self, frame: np.ndarray, stats: dict, header_lines: list) -> None:
        if self.on_frame:
            self.on_frame(frame_to_jpeg_bytes(frame), stats, header_lines)

    def process_disabled(self, frame: np.ndarray, metadata: dict) -> tuple[np.ndarray, dict]:
        return frame, {"disabled": True}

    @abstractmethod
    def process(self, frame: np.ndarray, metadata: dict) -> tuple[np.ndarray, dict]:
        pass

    @abstractmethod
    def build_header_lines(self, stats: dict, metadata: dict) -> list[str]:
        pass

    def _fps_line(self) -> str:
        return f"FPS: {self._fps_counter.fps:.1f}"

    def _source_line(self, metadata: dict) -> str:
        return metadata.get("source_label", "unknown")
