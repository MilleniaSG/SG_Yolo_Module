import time
from collections import deque
from typing import Optional


class FpsCounter:
    def __init__(self, window_size: int = 30):
        self._window_size = window_size
        self._timestamps: deque = deque(maxlen=window_size)
        self._last_time: Optional[float] = None

    def tick(self) -> None:
        now = time.perf_counter()
        if self._last_time is not None:
            self._timestamps.append(now - self._last_time)
        self._last_time = now

    @property
    def fps(self) -> float:
        if not self._timestamps:
            return 0.0
        avg_delta = sum(self._timestamps) / len(self._timestamps)
        if avg_delta <= 0:
            return 0.0
        return 1.0 / avg_delta

    def reset(self) -> None:
        self._timestamps.clear()
        self._last_time = None
