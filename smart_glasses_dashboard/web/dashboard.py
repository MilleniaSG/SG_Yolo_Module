"""Dashboard orchestration for the web server."""

from __future__ import annotations

import base64
import queue
import threading
from typing import Any, Optional

from panels.unified_processor import UnifiedProcessor
from source_manager import SPEED_OPTIONS, SourceManager, SourceMode
from utils.camera_detect import enumerate_cameras, get_default_iphone_index
from utils.fusion_state import fusion_state

PANEL_WIDTH = 640
PANEL_HEIGHT = 480
MAX_PC_POINTS_WS = 8000


class Dashboard:
    def __init__(self):
        self.lock = threading.Lock()
        self._queues = [queue.Queue(maxsize=1)]
        self._playback = {
            "playing": False,
            "current_sec": 0.0,
            "total_sec": 0.0,
            "source": "",
            "speed_index": 3,
            "loop": False,
            "error": "",
        }
        self._panels: dict[str, dict[str, Any]] = {
            "original": {"jpeg": b"", "stats": {}, "header": []},
            "yolo": {"jpeg": b"", "stats": {}, "header": []},
            "orb": {"jpeg": b"", "stats": {}, "header": []},
        }
        self._processor: Optional[UnifiedProcessor] = None
        self._source_manager: Optional[SourceManager] = None
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True

        self._processor = UnifiedProcessor(
            self._queues[0],
            on_original=lambda j, s, h: self._update_panel("original", j, s, h),
            on_yolo=lambda j, s, h: self._update_panel("yolo", j, s, h),
            on_orb=lambda j, s, h: self._update_panel("orb", j, s, h),
            panel_width=PANEL_WIDTH,
            panel_height=PANEL_HEIGHT,
        )

        self._source_manager = SourceManager(
            self._queues,
            on_source_changed=self._on_source_changed,
            on_playback_changed=self._on_playback_changed,
            on_position_changed=self._on_position_changed,
            on_error=self._on_error,
        )

        self._processor.start()
        self._source_manager.switch_to_webcam(0)

    def _update_panel(self, key: str, jpeg: bytes, stats: dict, header: list) -> None:
        with self.lock:
            self._panels[key] = {"jpeg": jpeg, "stats": stats, "header": header}

    def _on_source_changed(self, label: str) -> None:
        with self.lock:
            self._playback["source"] = label
            self._playback["error"] = ""

    def _on_playback_changed(self, playing: bool) -> None:
        with self.lock:
            self._playback["playing"] = playing

    def _on_position_changed(self, current: float, total: float) -> None:
        with self.lock:
            self._playback["current_sec"] = current
            self._playback["total_sec"] = total

    def _on_error(self, message: str) -> None:
        with self.lock:
            self._playback["error"] = message

    def _pointcloud_payload(self) -> dict:
        with fusion_state.lock:
            pose = fusion_state.pose
            cloud = fusion_state.cloud
            n = len(cloud.points)
            if n == 0:
                return {
                    "points": [],
                    "colors": [],
                    "trajectory": [],
                    "position": [0.0, 0.0, 0.0],
                    "yaw_deg": 0.0,
                    "status": "INIT",
                    "count": 0,
                    "distance_m": 0.0,
                }
            step = max(1, n // MAX_PC_POINTS_WS)
            pts = cloud.points[::step].tolist()
            cols = [
                [int(c[2]), int(c[1]), int(c[0])]  # BGR -> RGB for Three.js
                for c in cloud.colors[::step]
            ]
            traj = [t.tolist() for t in (pose.trajectory if pose else [])]
            pos = pose.t_wc.tolist() if pose else [0.0, 0.0, 0.0]
            return {
                "points": pts,
                "colors": cols,
                "trajectory": traj,
                "position": pos,
                "yaw_deg": pose.yaw_deg if pose else 0.0,
                "status": pose.status if pose else "INIT",
                "count": n,
                "distance_m": pose.distance_m if pose else 0.0,
            }

    def get_state(self) -> dict:
        with self.lock:
            panels = {}
            for key, data in self._panels.items():
                jpeg = data["jpeg"]
                panels[key] = {
                    "image": base64.b64encode(jpeg).decode("ascii") if jpeg else "",
                    "stats": data["stats"],
                    "header": data["header"],
                }
            return {
                "panels": panels,
                "playback": dict(self._playback),
                "speed_options": SPEED_OPTIONS,
                "pointcloud": self._pointcloud_payload(),
            }

    def get_cameras(self) -> list[dict]:
        return enumerate_cameras()

    def toggle_play(self) -> None:
        if self._source_manager:
            self._source_manager.toggle_play()

    def set_speed_index(self, index: int) -> None:
        if self._source_manager:
            self._source_manager.set_speed_index(index)
        with self.lock:
            self._playback["speed_index"] = index

    def set_loop(self, enabled: bool) -> None:
        if self._source_manager:
            self._source_manager.set_loop(enabled)
        with self.lock:
            self._playback["loop"] = enabled

    def seek(self, fraction: float) -> None:
        if self._source_manager and self._source_manager.mode == SourceMode.FILE:
            self._source_manager.seek(fraction * self._source_manager.duration_sec)
            self._reset()

    def switch_webcam(self, index: int = 0) -> None:
        self._reset()
        if self._source_manager:
            self._source_manager.switch_to_webcam(index)

    def switch_continuity(self, index: Optional[int] = None) -> None:
        if index is None:
            index = get_default_iphone_index()
            if index is None:
                cameras = enumerate_cameras()
                index = cameras[0]["index"] if cameras else 0
        self._reset()
        if self._source_manager:
            self._source_manager.switch_to_continuity(index)

    def switch_file(self, path: str) -> None:
        self._reset()
        if self._source_manager:
            self._source_manager.switch_to_file(path, resume=True)

    def toggle_yolo(self) -> bool:
        if self._processor:
            self._processor.set_yolo_enabled(not self._processor.yolo_enabled)
            return self._processor.yolo_enabled
        return False

    def toggle_orb(self) -> bool:
        if self._processor:
            self._processor.set_orb_enabled(not self._processor.orb_enabled)
            return self._processor.orb_enabled
        return False

    def _reset(self) -> None:
        if self._processor:
            self._processor.reset_state()

    def stop(self) -> None:
        if self._source_manager:
            self._source_manager.stop()
        if self._processor:
            self._processor.stop()


dashboard = Dashboard()
