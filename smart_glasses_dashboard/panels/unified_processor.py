"""Single-frame processor: YOLO 3D, ORB-SLAM viz, and point-cloud building."""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from typing import Callable, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from utils.depth_estimator import DepthEstimator
from utils.device import get_inference_device, use_half_precision
from utils.frame_utils import (
    COLOR_ORB,
    COLOR_ORIGINAL,
    COLOR_YOLO,
    class_color,
    draw_header_bar_inplace,
    frame_to_jpeg_bytes,
    make_placeholder,
    sample_pixel_bgr,
)
from utils.fusion_state import fusion_state
from utils.point_cloud import DEPTH_COLOR_POINTS_PER_FRAME, ORB_POINTS_PER_FRAME
from utils.pose_tracker import PoseTracker
from utils.yolo3d import (
    camera_intrinsics,
    draw_3d_cuboid,
    draw_depth_axis,
    estimate_3d_box_camera_corners,
    unproject_pixel,
)

FrameCallback = Callable[[bytes, dict, list], None]

YOLO_IMGSZ = 320
YOLO_MAX_DET = 20
MATCH_DRAW_LIMIT = 25


class UnifiedProcessor(threading.Thread):
    """Processes each frame once; emits original, YOLO, and ORB panel JPEGs."""

    def __init__(
        self,
        input_queue: queue.Queue,
        on_original: Optional[FrameCallback] = None,
        on_yolo: Optional[FrameCallback] = None,
        on_orb: Optional[FrameCallback] = None,
        panel_width: int = 640,
        panel_height: int = 480,
    ):
        super().__init__(daemon=True)
        self.input_queue = input_queue
        self.on_original = on_original
        self.on_yolo = on_yolo
        self.on_orb = on_orb
        self.panel_width = panel_width
        self.panel_height = panel_height
        self._running = True
        self._yolo_enabled = True
        self._orb_enabled = True
        self._fps_orig = self._fps_yolo = self._fps_orb = 0.0
        self._fps_times: dict[str, list] = {"orig": [], "yolo": [], "orb": []}

        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        os.makedirs(model_dir, exist_ok=True)
        self._device = get_inference_device()
        self._half = use_half_precision(self._device)
        self._model = YOLO(os.path.join(model_dir, "yolov8n.pt"))
        self._depth = DepthEstimator()
        self._pose = PoseTracker()
        fusion_state.pose = self._pose

        warmup = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        for _ in range(2):
            self._model.predict(
                warmup, imgsz=YOLO_IMGSZ, device=self._device,
                half=self._half, max_det=YOLO_MAX_DET, verbose=False,
            )
        if self._depth.ready:
            self._depth.estimate(warmup)

    def set_yolo_enabled(self, enabled: bool) -> None:
        self._yolo_enabled = enabled

    def set_orb_enabled(self, enabled: bool) -> None:
        self._orb_enabled = enabled

    @property
    def yolo_enabled(self) -> bool:
        return self._yolo_enabled

    @property
    def orb_enabled(self) -> bool:
        return self._orb_enabled

    def reset_state(self) -> None:
        self._pose.reset()
        fusion_state.reset()
        fusion_state.pose = self._pose

    def stop(self) -> None:
        self._running = False

    def _tick_fps(self, key: str) -> float:
        now = time.perf_counter()
        times = self._fps_times[key]
        times.append(now)
        if len(times) > 30:
            times.pop(0)
        if len(times) < 2:
            return 0.0
        return (len(times) - 1) / (times[-1] - times[0])

    def _emit(self, cb: Optional[FrameCallback], frame: np.ndarray, stats: dict, header: list) -> None:
        if not cb:
            return
        h, w = frame.shape[:2]
        if (w, h) != (self.panel_width, self.panel_height):
            interp = cv2.INTER_AREA if w > self.panel_width else cv2.INTER_LINEAR
            frame = cv2.resize(frame, (self.panel_width, self.panel_height), interpolation=interp)
        cb(frame_to_jpeg_bytes(frame), stats, header)

    def run(self) -> None:
        for name, color, cb in [
            ("ORIGINAL FEED", COLOR_ORIGINAL, self.on_original),
            ("YOLO 3D", COLOR_YOLO, self.on_yolo),
            ("ORB-SLAM", COLOR_ORB, self.on_orb),
        ]:
            ph = make_placeholder(self.panel_width, self.panel_height, name, color)
            self._emit(cb, ph, {}, [f"FPS: 0.0"])

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
                self._process_frame(frame, metadata)
            except Exception as exc:
                logging.getLogger(__name__).warning("Frame processing failed: %s", exc)

    def _process_frame(self, frame: np.ndarray, metadata: dict) -> None:
        source = metadata.get("source_label", "unknown")

        # --- Original ---
        orig_stats = {
            "resolution": metadata.get("resolution", ""),
            "frame_number": metadata.get("frame_number", 0),
        }
        orig_header = [
            f"FPS: {self._tick_fps('orig'):.1f}",
            f"Res: {orig_stats['resolution']}",
            f"Frame: {orig_stats['frame_number']}",
            source,
        ]
        orig_disp = draw_header_bar_inplace(
            frame.copy(), "ORIGINAL FEED", COLOR_ORIGINAL, orig_header
        )
        self._emit(self.on_original, orig_disp, orig_stats, orig_header)

        if not self._yolo_enabled and not self._orb_enabled:
            return

        h, w = frame.shape[:2]
        fx, fy, cx, cy = camera_intrinsics(w, h)
        depth_map = self._depth.estimate(frame)
        kp, matches, prev_kp = self._pose.process(
            frame, depth_map, self._depth, fx, fy, cx, cy
        )

        new_points, new_colors = [], []
        if depth_map is not None:
            step = max(4, int(np.sqrt(h * w / DEPTH_COLOR_POINTS_PER_FRAME)))
            added = 0
            for v in range(step // 2, h, step):
                for u in range(step // 2, w, step):
                    if added >= DEPTH_COLOR_POINTS_PER_FRAME:
                        break
                    d = self._depth.depth_at_pixel(depth_map, u, v, radius=2)
                    if d is None or d < 0.35 or d > 25.0:
                        continue
                    p_cam = unproject_pixel(float(u), float(v), d, fx, fy, cx, cy)
                    p_world = self._pose.camera_to_world(p_cam.reshape(1, 3))[0]
                    new_points.append(p_world)
                    new_colors.append(sample_pixel_bgr(frame, u, v))
                    added += 1
                if added >= DEPTH_COLOR_POINTS_PER_FRAME:
                    break

        if kp:
            step = max(1, len(kp) // ORB_POINTS_PER_FRAME)
            for i in range(0, len(kp), step):
                u, v = kp[i].pt
                d = self._depth.depth_at_pixel(depth_map, int(u), int(v), radius=3)
                if d is None or d < 0.4:
                    continue
                p_cam = unproject_pixel(u, v, d, fx, fy, cx, cy)
                p_world = self._pose.camera_to_world(p_cam.reshape(1, 3))[0]
                new_points.append(p_world)
                new_colors.append(sample_pixel_bgr(frame, u, v))

        # --- YOLO 3D ---
        yolo_out = frame.copy()
        object_count = 0
        inference_ms = 0.0
        if self._yolo_enabled:
            t0 = time.perf_counter()
            result = self._model.predict(
                frame, imgsz=YOLO_IMGSZ, device=self._device,
                half=self._half, max_det=YOLO_MAX_DET, verbose=False,
            )[0]
            inference_ms = (time.perf_counter() - t0) * 1000.0

            if result.boxes is not None and len(result.boxes) > 0:
                for box, conf, cls_id in zip(
                    result.boxes.xyxy.cpu().numpy(),
                    result.boxes.conf.cpu().numpy(),
                    result.boxes.cls.cpu().numpy().astype(int),
                ):
                    name = result.names[int(cls_id)]
                    color = class_color(int(cls_id))
                    x1, y1, x2, y2 = map(int, box)
                    depth_override = self._depth.depth_at_box(depth_map, x1, y1, x2, y2)
                    cam_corners, corners_2d, depth_m = estimate_3d_box_camera_corners(
                        float(x1), float(y1), float(x2), float(y2),
                        name, frame.shape[:2], depth_override=depth_override,
                    )
                    world_corners = self._pose.camera_to_world(cam_corners)
                    for p_w, corner_2d in zip(world_corners, corners_2d):
                        new_points.append(p_w.astype(np.float32))
                        new_colors.append(sample_pixel_bgr(frame, corner_2d[0], corner_2d[1]))
                    draw_3d_cuboid(
                        yolo_out, corners_2d, color,
                        label=f"{name} {conf:.2f} | {depth_m:.1f}m",
                    )
                    object_count += 1
            draw_depth_axis(yolo_out)
            pos = self._pose.t_wc
            cv2.putText(yolo_out, f"Pos X:{pos[0]:+.1f} Z:{pos[2]:+.1f}m",
                        (8, 52), cv2.FONT_HERSHEY_DUPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)

        with fusion_state.lock:
            if new_points:
                fusion_state.cloud.add_points(
                    np.array(new_points, dtype=np.float32),
                    np.array(new_colors, dtype=np.uint8),
                )
            fusion_state.object_count = object_count
            fusion_state.depth_ready = self._depth.ready
            fusion_state.inference_ms = inference_ms

        yolo_stats = {
            "object_count": object_count,
            "inference_ms": inference_ms,
            "status": self._pose.status,
            "pos_x": float(self._pose.t_wc[0]),
            "pos_z": float(self._pose.t_wc[2]),
            "cloud_points": fusion_state.cloud.count,
            "disabled": not self._yolo_enabled,
        }
        yolo_header = (
            [f"FPS: {self._tick_fps('yolo'):.1f}", "YOLO OFF", source]
            if not self._yolo_enabled
            else [
                f"FPS: {self._tick_fps('yolo'):.1f}",
                self._pose.status,
                f"Obj:{object_count}",
                f"Pts:{fusion_state.cloud.count}",
                f"{inference_ms:.0f}ms",
                source,
            ]
        )
        yolo_disp = draw_header_bar_inplace(yolo_out, "YOLO 3D", COLOR_YOLO, yolo_header)
        self._emit(self.on_yolo, yolo_disp, yolo_stats, yolo_header)

        # --- ORB-SLAM ---
        orb_out = frame.copy()
        if self._orb_enabled and kp:
            cv2.drawKeypoints(
                orb_out, kp, orb_out, color=(0, 255, 255),
                flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT,
            )
        if self._orb_enabled and prev_kp is not None and kp:
            for m in matches[:MATCH_DRAW_LIMIT]:
                pt1 = tuple(map(int, prev_kp[m.queryIdx].pt))
                pt2 = tuple(map(int, kp[m.trainIdx].pt))
                cv2.line(orb_out, pt1, pt2, (0, 200, 255), 1, cv2.LINE_AA)

        pos = self._pose.t_wc
        orb_stats = {
            "keypoint_count": self._pose.keypoint_count,
            "match_count": self._pose.match_count,
            "status": self._pose.status,
            "pos_x": float(pos[0]),
            "pos_y": float(pos[1]),
            "pos_z": float(pos[2]),
            "distance_m": self._pose.distance_m,
            "yaw_deg": self._pose.yaw_deg,
            "disabled": not self._orb_enabled,
        }
        orb_header = (
            [f"FPS: {self._tick_fps('orb'):.1f}", "ORB OFF", source]
            if not self._orb_enabled
            else [
                f"FPS: {self._tick_fps('orb'):.1f}",
                f"KP:{self._pose.keypoint_count}",
                f"Match:{self._pose.match_count}",
                self._pose.status,
                f"X:{pos[0]:+.1f} Z:{pos[2]:+.1f}m",
                source,
            ]
        )
        orb_disp = draw_header_bar_inplace(orb_out, "ORB-SLAM", COLOR_ORB, orb_header)
        self._emit(self.on_orb, orb_disp, orb_stats, orb_header)
