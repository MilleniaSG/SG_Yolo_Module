"""Accumulating world-frame point cloud and map rendering."""

from __future__ import annotations

import cv2
import numpy as np

MAX_POINTS = 60000
ORB_POINTS_PER_FRAME = 80
DEPTH_COLOR_POINTS_PER_FRAME = 120
VOXEL_SIZE = 0.12


class PointCloudMap:
    def __init__(self):
        self.points = np.zeros((0, 3), dtype=np.float32)
        self.colors = np.zeros((0, 3), dtype=np.uint8)
        self._voxel_cache: dict[tuple[int, int, int], int] = {}

    def reset(self) -> None:
        self.points = np.zeros((0, 3), dtype=np.float32)
        self.colors = np.zeros((0, 3), dtype=np.uint8)
        self._voxel_cache.clear()

    @property
    def count(self) -> int:
        return len(self.points)

    def add_points(self, points_world: np.ndarray, colors: np.ndarray) -> None:
        if points_world.size == 0:
            return
        points_world = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
        colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)

        new_pts = []
        new_cols = []
        for p, c in zip(points_world, colors):
            key = (
                int(p[0] / VOXEL_SIZE),
                int(p[1] / VOXEL_SIZE),
                int(p[2] / VOXEL_SIZE),
            )
            if key in self._voxel_cache:
                continue
            self._voxel_cache[key] = 1
            new_pts.append(p)
            new_cols.append(c)

        if not new_pts:
            return

        add_p = np.array(new_pts, dtype=np.float32)
        add_c = np.array(new_cols, dtype=np.uint8)
        self.points = np.vstack([self.points, add_p]) if len(self.points) else add_p
        self.colors = np.vstack([self.colors, add_c]) if len(self.colors) else add_c

        if len(self.points) > MAX_POINTS:
            trim = len(self.points) - MAX_POINTS
            self.points = self.points[trim:]
            self.colors = self.colors[trim:]

    def render_bev(
        self,
        width: int,
        height: int,
        trajectory: list[np.ndarray],
        current_pos: np.ndarray,
        yaw_deg: float,
    ) -> np.ndarray:
        canvas = np.full((height, width, 3), 13, dtype=np.uint8)
        margin = 36
        plot_w = width - 2 * margin
        plot_h = height - 2 * margin

        if len(self.points) == 0 and len(trajectory) <= 1:
            cv2.putText(
                canvas,
                "POINT CLOUD MAP",
                (margin, margin + 20),
                cv2.FONT_HERSHEY_DUPLEX,
                0.55,
                (0, 200, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                "Move camera to build map",
                (margin, margin + 44),
                cv2.FONT_HERSHEY_DUPLEX,
                0.4,
                (140, 140, 140),
                1,
                cv2.LINE_AA,
            )
            return canvas

        xs = self.points[:, 0] if len(self.points) else np.array([0.0])
        zs = self.points[:, 2] if len(self.points) else np.array([0.0])
        traj = np.array(trajectory) if trajectory else np.zeros((1, 3))
        all_x = np.concatenate([xs, traj[:, 0]])
        all_z = np.concatenate([zs, traj[:, 2]])
        cx = (all_x.min() + all_x.max()) * 0.5
        cz = (all_z.min() + all_z.max()) * 0.5
        span = max(all_x.max() - all_x.min(), all_z.max() - all_z.min(), 1.0)
        scale = min(plot_w, plot_h) * 0.42 / span

        def to_px(x: float, z: float) -> tuple[int, int]:
            px = int(margin + plot_w * 0.5 + (x - cx) * scale)
            py = int(margin + plot_h * 0.5 + (z - cz) * scale)
            return px, py

        if len(self.points):
            ys = self.points[:, 1]
            y_span = max(ys.max() - ys.min(), 0.5)
            for p, c in zip(self.points, self.colors):
                px, py = to_px(float(p[0]), float(p[2]))
                if margin <= px < width - margin and margin <= py < height - margin:
                    height_factor = 1.0 - (float(p[1]) - ys.min()) / y_span
                    radius = max(1, int(1 + height_factor * 2))
                    color = tuple(int(v) for v in c)
                    cv2.circle(canvas, (px, py), radius, color, -1, cv2.LINE_AA)

        if len(trajectory) >= 2:
            prev = None
            for t in trajectory:
                pt = to_px(float(t[0]), float(t[2]))
                if prev is not None:
                    cv2.line(canvas, prev, pt, (0, 200, 255), 2, cv2.LINE_AA)
                prev = pt

        cam_px = to_px(float(current_pos[0]), float(current_pos[2]))
        cv2.circle(canvas, cam_px, 7, (0, 255, 255), -1, cv2.LINE_AA)
        yaw_rad = np.radians(yaw_deg)
        tip = (
            int(cam_px[0] + 18 * np.sin(yaw_rad)),
            int(cam_px[1] + 18 * np.cos(yaw_rad)),
        )
        cv2.arrowedLine(canvas, cam_px, tip, (255, 255, 255), 2, tipLength=0.35)

        cv2.rectangle(
            canvas,
            (margin - 2, margin - 2),
            (width - margin + 2, height - margin + 2),
            (40, 40, 60),
            1,
        )
        cv2.putText(
            canvas,
            "TOP-DOWN MAP  (X-Z plane)",
            (margin, 22),
            cv2.FONT_HERSHEY_DUPLEX,
            0.42,
            (0, 200, 255),
            1,
            cv2.LINE_AA,
        )
        return canvas

    def project_to_frame(
        self,
        frame: np.ndarray,
        R_wc: np.ndarray,
        t_wc: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        max_dots: int = 400,
    ) -> None:
        if len(self.points) == 0:
            return

        R_cw = R_wc.T
        step = max(1, len(self.points) // max_dots)
        h, w = frame.shape[:2]

        for i in range(0, len(self.points), step):
            p_w = self.points[i]
            p_c = R_cw @ (p_w - t_wc)
            z = float(p_c[2])
            if z < 0.25:
                continue
            u = int(fx * p_c[0] / z + cx)
            v = int(fy * p_c[1] / z + cy)
            if 0 <= u < w and 0 <= v < h:
                c = tuple(int(x) for x in self.colors[i])
                cv2.circle(frame, (u, v), 2, c, -1, cv2.LINE_AA)
