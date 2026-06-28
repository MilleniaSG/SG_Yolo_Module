"""Thread-safe shared state between fusion video and map panels."""

from __future__ import annotations

import threading
from typing import Optional

from utils.point_cloud import PointCloudMap
from utils.pose_tracker import PoseTracker


class FusionState:
    def __init__(self):
        self.lock = threading.Lock()
        self.pose: Optional[PoseTracker] = None
        self.cloud = PointCloudMap()
        self.object_count = 0
        self.inference_ms = 0.0
        self.depth_ready = False

    def reset(self) -> None:
        with self.lock:
            if self.pose is not None:
                self.pose.reset()
            self.cloud.reset()
            self.object_count = 0
            self.inference_ms = 0.0


fusion_state = FusionState()
