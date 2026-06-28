import os
import time
from typing import Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from panels.base_panel import PanelWorker
from utils.device import get_inference_device, use_half_precision
from utils.frame_utils import COLOR_YOLO, class_color
from utils.yolo3d import (
    draw_3d_cuboid,
    draw_depth_axis,
    estimate_3d_box,
)

YOLO_IMGSZ = 320
YOLO_MAX_DET = 20
DEPTH_SIZE = 192


class YoloPanel(PanelWorker):
    def __init__(self, input_queue, panel_width=640, panel_height=480):
        super().__init__(
            "YOLO 3D",
            COLOR_YOLO,
            input_queue,
            panel_width,
            panel_height,
        )
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "yolov8n.pt")
        self._device = get_inference_device()
        self._half = use_half_precision(self._device)
        self._torch_device = torch.device(self._device)
        self._model = YOLO(model_path)

        self._depth_model = None
        self._depth_transform = None
        self._depth_ready = False
        self._init_depth_model()

        warmup = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        for _ in range(3):
            self._model.predict(
                warmup,
                imgsz=YOLO_IMGSZ,
                device=self._device,
                half=self._half,
                max_det=YOLO_MAX_DET,
                verbose=False,
            )
        if self._depth_ready:
            self._estimate_depth_map(warmup)

    def _init_depth_model(self) -> None:
        try:
            self._depth_model = torch.hub.load(
                "intel-isl/MiDaS", "MiDaS_small", trust_repo=True
            )
            self._depth_model.to(self._torch_device).eval()
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self._depth_transform = transforms.small_transform
            self._depth_ready = True
        except Exception:
            self._depth_model = None
            self._depth_transform = None
            self._depth_ready = False

    def _estimate_depth_map(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if not self._depth_ready:
            return None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (DEPTH_SIZE, DEPTH_SIZE), interpolation=cv2.INTER_LINEAR)
        sample = self._depth_transform(small)
        if isinstance(sample, dict):
            tensor = sample["image"]
        else:
            tensor = sample

        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self._torch_device)

        with torch.no_grad():
            prediction = self._depth_model(tensor)
            depth = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(frame.shape[0], frame.shape[1]),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_np = depth.float().cpu().numpy()
        depth_np = depth_np - depth_np.min()
        depth_np = depth_np / max(depth_np.max(), 1e-6)
        return depth_np

    def _depth_at_box(self, depth_map: Optional[np.ndarray], x1: int, y1: int, x2: int, y2: int) -> Optional[float]:
        if depth_map is None:
            return None
        h, w = depth_map.shape[:2]
        cx = int(np.clip((x1 + x2) * 0.5, 0, w - 1))
        cy = int(np.clip((y1 + y2) * 0.5, 0, h - 1))
        radius = max(2, int(min(x2 - x1, y2 - y1) * 0.15))
        y0 = max(0, cy - radius)
        y1c = min(h, cy + radius + 1)
        x0 = max(0, cx - radius)
        x1c = min(w, cx + radius + 1)
        patch = depth_map[y0:y1c, x0:x1c]
        if patch.size == 0:
            return None
        relative = float(np.median(patch))
        # Map relative inverse-depth to meters (closer = higher relative value).
        return 1.5 + (1.0 - relative) * 14.0

    def process(self, frame: np.ndarray, metadata: dict) -> tuple[np.ndarray, dict]:
        start = time.perf_counter()
        results = self._model.predict(
            frame,
            imgsz=YOLO_IMGSZ,
            device=self._device,
            half=self._half,
            max_det=YOLO_MAX_DET,
            verbose=False,
        )
        result = results[0]
        yolo_ms = (time.perf_counter() - start) * 1000.0

        out = frame
        detected_classes: dict[str, tuple[int, int, int]] = {}
        object_count = 0
        depths: list[float] = []

        if result.boxes is not None and len(result.boxes) > 0:
            out = frame.copy()
            depth_map = self._estimate_depth_map(frame)

            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confs, cls_ids):
                name = result.names[int(cls_id)]
                color = class_color(int(cls_id))
                detected_classes[name] = color
                x1, y1, x2, y2 = map(int, box)

                depth_override = self._depth_at_box(depth_map, x1, y1, x2, y2)
                corners, depth_m = estimate_3d_box(
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                    name,
                    frame.shape[:2],
                    depth_override=depth_override,
                )
                depths.append(depth_m)

                label = f"{name} {conf:.2f} | {depth_m:.1f}m"
                draw_3d_cuboid(out, corners, color, label=label, thickness=2)
                object_count += 1

            draw_depth_axis(out)
            self._draw_legend(out, detected_classes)

        inference_ms = (time.perf_counter() - start) * 1000.0
        avg_depth = float(np.mean(depths)) if depths else 0.0

        return out, {
            "object_count": object_count,
            "inference_ms": inference_ms,
            "yolo_ms": yolo_ms,
            "avg_depth_m": avg_depth,
            "mode_3d": self._depth_ready,
        }

    def process_disabled(self, frame: np.ndarray, metadata: dict) -> tuple[np.ndarray, dict]:
        return frame, {"disabled": True, "object_count": 0, "inference_ms": 0.0}

    def build_header_lines(self, stats: dict, metadata: dict) -> list[str]:
        if stats.get("disabled"):
            return [
                self._fps_line(),
                "YOLO 3D OFF",
                self._source_line(metadata),
            ]
        depth_tag = f"{stats.get('avg_depth_m', 0.0):.1f}m avg"
        mode = "MiDaS+3D" if stats.get("mode_3d") else "Mono3D"
        return [
            self._fps_line(),
            mode,
            f"Objects: {stats.get('object_count', 0)}",
            depth_tag,
            f"{stats.get('inference_ms', 0.0):.1f}ms",
            self._source_line(metadata),
        ]

    def _draw_legend(
        self,
        frame: np.ndarray,
        detected_classes: dict[str, tuple[int, int, int]],
    ) -> None:
        if not detected_classes:
            return

        h, w = frame.shape[:2]
        items = list(detected_classes.items())[:6]
        line_h = 16
        box_w = 130
        box_h = 6 + line_h * len(items)
        x0 = w - box_w - 8
        y0 = 42

        cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), (20, 20, 20), -1)
        cv2.putText(
            frame,
            "3D classes",
            (x0 + 6, y0 + 12),
            cv2.FONT_HERSHEY_DUPLEX,
            0.35,
            (180, 255, 120),
            1,
            cv2.LINE_AA,
        )

        for i, (name, color) in enumerate(items):
            y = y0 + 26 + i * line_h
            cv2.rectangle(frame, (x0 + 4, y - 7), (x0 + 14, y + 1), color, -1)
            cv2.putText(
                frame,
                name[:12],
                (x0 + 18, y),
                cv2.FONT_HERSHEY_DUPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
