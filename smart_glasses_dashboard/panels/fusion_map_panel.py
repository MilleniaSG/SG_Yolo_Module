"""Point-cloud map panel with camera localization (reads shared fusion state)."""

import cv2
import numpy as np

from panels.base_panel import PanelWorker
from utils.frame_utils import COLOR_ORB
from utils.fusion_state import fusion_state


class FusionMapPanel(PanelWorker):
    def __init__(self, input_queue, on_frame=None, panel_width=640, panel_height=480):
        super().__init__(
            "POINT CLOUD MAP",
            COLOR_ORB,
            input_queue,
            on_frame=on_frame,
            panel_width=panel_width,
            panel_height=panel_height,
        )

    def reset_state(self) -> None:
        fusion_state.reset()

    def process(self, frame: np.ndarray, metadata: dict) -> tuple[np.ndarray, dict]:
        with fusion_state.lock:
            pose = fusion_state.pose
            cloud = fusion_state.cloud
            if pose is None:
                map_img = cloud.render_bev(
                    frame.shape[1], frame.shape[0], [], np.zeros(3), 0.0
                )
                stats = {"cloud_points": 0, "status": "INIT", "match_count": 0,
                         "keypoint_count": 0, "pos_x": 0, "pos_y": 0, "pos_z": 0,
                         "distance_m": 0, "yaw_deg": 0, "object_count": 0}
            else:
                map_img = cloud.render_bev(
                    frame.shape[1],
                    frame.shape[0],
                    pose.trajectory,
                    pose.t_wc,
                    pose.yaw_deg,
                )
                pos = pose.t_wc.copy()
                stats = {
                    "cloud_points": cloud.count,
                    "status": pose.status,
                    "match_count": pose.match_count,
                    "keypoint_count": pose.keypoint_count,
                    "pos_x": float(pos[0]),
                    "pos_y": float(pos[1]),
                    "pos_z": float(pos[2]),
                    "distance_m": pose.distance_m,
                    "yaw_deg": pose.yaw_deg,
                    "object_count": fusion_state.object_count,
                }

        self._draw_localization_overlay(map_img, stats)
        return map_img, stats

    def process_disabled(self, frame: np.ndarray, metadata: dict) -> tuple[np.ndarray, dict]:
        out = np.full_like(frame, 13)
        cv2.putText(
            out,
            "MAP OFF",
            (frame.shape[1] // 2 - 40, frame.shape[0] // 2),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            COLOR_ORB,
            2,
            cv2.LINE_AA,
        )
        return out, {"disabled": True}

    def build_header_lines(self, stats: dict, metadata: dict) -> list[str]:
        if stats.get("disabled"):
            return [self._fps_line(), "MAP OFF", self._source_line(metadata)]
        return [
            self._fps_line(),
            stats.get("status", "LOST"),
            f"Pts:{stats.get('cloud_points', 0)}",
            f"X:{stats.get('pos_x', 0):+.1f} Y:{stats.get('pos_y', 0):+.1f} Z:{stats.get('pos_z', 0):+.1f}",
            f"Path:{stats.get('distance_m', 0):.1f}m",
        ]

    def _draw_localization_overlay(self, frame: np.ndarray, stats: dict) -> None:
        box_h = 72
        y0 = frame.shape[0] - box_h - 10
        cv2.rectangle(frame, (8, y0), (frame.shape[1] - 8, y0 + box_h), (18, 18, 28), -1)
        cv2.rectangle(frame, (8, y0), (frame.shape[1] - 8, y0 + box_h), COLOR_ORB, 1)
        lines = [
            "LOCALIZATION",
            f"Position  X:{stats['pos_x']:+.2f}m  Y:{stats['pos_y']:+.2f}m  Z:{stats['pos_z']:+.2f}m",
            f"Heading {stats['yaw_deg']:+.1f} deg   Distance traveled {stats['distance_m']:.2f}m",
            f"Map points {stats['cloud_points']}   ORB matches {stats['match_count']}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (16, y0 + 18 + i * 16),
                cv2.FONT_HERSHEY_DUPLEX,
                0.38,
                (0, 255, 255) if i == 0 else (210, 210, 210),
                1,
                cv2.LINE_AA,
            )
