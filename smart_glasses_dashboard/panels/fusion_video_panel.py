"""Fused YOLO 3D + ORB-SLAM video panel with live point-cloud overlay."""

import os
import time

import cv2
import numpy as np
from ultralytics import YOLO

from panels.base_panel import PanelWorker
from utils.depth_estimator import DepthEstimator
from utils.device import get_inference_device, use_half_precision
from utils.frame_utils import COLOR_YOLO, class_color, sample_pixel_bgr
from utils.fusion_state import fusion_state
from utils.point_cloud import ORB_POINTS_PER_FRAME
from utils.pose_tracker import PoseTracker
from utils.yolo3d import (
    camera_intrinsics,
    draw_3d_cuboid,
    draw_depth_axis,
    estimate_3d_box_camera_corners,
    unproject_pixel,
)

YOLO_IMGSZ = 320
YOLO_MAX_DET = 20


class FusionVideoPanel(PanelWorker):
    def __init__(self, input_queue, on_frame=None, panel_width=640, panel_height=480):
        super().__init__(
            "YOLO+ORB FUSION",
            COLOR_YOLO,
            input_queue,
            on_frame=on_frame,
            panel_width=panel_width,
            panel_height=panel_height,
        )
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "yolov8n.pt")
        self._device = get_inference_device()
        self._half = use_half_precision(self._device)
        self._model = YOLO(model_path)
        self._depth = DepthEstimator()
        self._pose = PoseTracker()
        fusion_state.pose = self._pose

        warmup = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        for _ in range(2):
            self._model.predict(
                warmup,
                imgsz=YOLO_IMGSZ,
                device=self._device,
                half=self._half,
                max_det=YOLO_MAX_DET,
                verbose=False,
            )
        if self._depth.ready:
            self._depth.estimate(warmup)

    def reset_state(self) -> None:
        self._pose.reset()
        fusion_state.reset()

    def process(self, frame: np.ndarray, metadata: dict) -> tuple[np.ndarray, dict]:
        start = time.perf_counter()
        h, w = frame.shape[:2]
        fx, fy, cx, cy = camera_intrinsics(w, h)

        depth_map = self._depth.estimate(frame)
        kp, matches, prev_kp = self._pose.process(
            frame, depth_map, self._depth, fx, fy, cx, cy
        )

        results = self._model.predict(
            frame,
            imgsz=YOLO_IMGSZ,
            device=self._device,
            half=self._half,
            max_det=YOLO_MAX_DET,
            verbose=False,
        )
        result = results[0]

        out = frame.copy()
        detected_classes: dict[str, tuple[int, int, int]] = {}
        object_count = 0
        new_points = []
        new_colors = []

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

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confs, cls_ids):
                name = result.names[int(cls_id)]
                color = class_color(int(cls_id))
                detected_classes[name] = color
                x1, y1, x2, y2 = map(int, box)

                depth_override = self._depth.depth_at_box(depth_map, x1, y1, x2, y2)
                cam_corners, corners_2d, depth_m = estimate_3d_box_camera_corners(
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                    name,
                    frame.shape[:2],
                    depth_override=depth_override,
                )
                world_corners = self._pose.camera_to_world(cam_corners)
                for p_w, corner_2d in zip(world_corners, corners_2d):
                    new_points.append(p_w.astype(np.float32))
                    new_colors.append(sample_pixel_bgr(frame, corner_2d[0], corner_2d[1]))

                label = f"{name} {conf:.2f} | {depth_m:.1f}m"
                draw_3d_cuboid(out, corners_2d, color, label=label, thickness=2)
                object_count += 1

        if kp:
            cv2.drawKeypoints(
                out,
                kp,
                out,
                color=(0, 255, 255),
                flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT,
            )

        if prev_kp is not None and kp:
            for m in matches:
                pt1 = tuple(map(int, prev_kp[m.queryIdx].pt))
                pt2 = tuple(map(int, kp[m.trainIdx].pt))
                cv2.line(out, pt1, pt2, (0, 200, 255), 1, cv2.LINE_AA)

        with fusion_state.lock:
            if new_points:
                fusion_state.cloud.add_points(
                    np.array(new_points, dtype=np.float32),
                    np.array(new_colors, dtype=np.uint8),
                )
            fusion_state.object_count = object_count
            fusion_state.depth_ready = self._depth.ready
            fusion_state.cloud.project_to_frame(
                out,
                self._pose.R_wc,
                self._pose.t_wc,
                fx,
                fy,
                cx,
                cy,
            )

        draw_depth_axis(out)
        self._draw_pose_hud(out)
        if detected_classes:
            self._draw_legend(out, detected_classes)

        inference_ms = (time.perf_counter() - start) * 1000.0
        with fusion_state.lock:
            fusion_state.inference_ms = inference_ms

        pos = self._pose.t_wc
        return out, {
            "object_count": object_count,
            "inference_ms": inference_ms,
            "keypoint_count": self._pose.keypoint_count,
            "match_count": self._pose.match_count,
            "status": self._pose.status,
            "pos_x": float(pos[0]),
            "pos_y": float(pos[1]),
            "pos_z": float(pos[2]),
            "distance_m": self._pose.distance_m,
            "yaw_deg": self._pose.yaw_deg,
            "cloud_points": fusion_state.cloud.count,
        }

    def process_disabled(self, frame: np.ndarray, metadata: dict) -> tuple[np.ndarray, dict]:
        return frame, {"disabled": True}

    def build_header_lines(self, stats: dict, metadata: dict) -> list[str]:
        if stats.get("disabled"):
            return [self._fps_line(), "FUSION OFF", self._source_line(metadata)]
        return [
            self._fps_line(),
            stats.get("status", "LOST"),
            f"Obj:{stats.get('object_count', 0)}",
            f"Pts:{stats.get('cloud_points', 0)}",
            f"Pos:{stats.get('pos_x', 0):.1f},{stats.get('pos_z', 0):.1f}m",
            f"{stats.get('inference_ms', 0):.0f}ms",
        ]

    def _draw_pose_hud(self, frame: np.ndarray) -> None:
        pos = self._pose.t_wc
        lines = [
            f"YOU ARE HERE",
            f"X:{pos[0]:+.2f}m  Y:{pos[1]:+.2f}m  Z:{pos[2]:+.2f}m",
            f"Yaw:{self._pose.yaw_deg:+.1f}  Path:{self._pose.distance_m:.2f}m",
        ]
        y0 = 52
        for i, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (8, y0 + i * 16),
                cv2.FONT_HERSHEY_DUPLEX,
                0.38,
                (0, 255, 255) if i == 0 else (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

    def _draw_legend(
        self,
        frame: np.ndarray,
        detected_classes: dict[str, tuple[int, int, int]],
    ) -> None:
        items = list(detected_classes.items())[:5]
        h, w = frame.shape[:2]
        x0 = w - 128
        y0 = 48
        cv2.rectangle(frame, (x0, y0), (w - 6, y0 + 10 + 14 * len(items)), (20, 20, 20), -1)
        for i, (name, color) in enumerate(items):
            y = y0 + 16 + i * 14
            cv2.rectangle(frame, (x0 + 4, y - 8), (x0 + 14, y), color, -1)
            cv2.putText(
                frame,
                name[:10],
                (x0 + 18, y),
                cv2.FONT_HERSHEY_DUPLEX,
                0.32,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
