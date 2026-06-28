"""Shared MiDaS monocular depth estimation."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch

from utils.device import get_inference_device

DEPTH_SIZE = 192


def relative_depth_to_meters(relative: float) -> float:
    return 1.5 + (1.0 - relative) * 14.0


class DepthEstimator:
    def __init__(self):
        self._device = get_inference_device()
        self._torch_device = torch.device(self._device)
        self._model = None
        self._transform = None
        self.ready = False
        self._init_model()

    def _init_model(self) -> None:
        try:
            self._model = torch.hub.load(
                "intel-isl/MiDaS", "MiDaS_small", trust_repo=True
            )
            self._model.to(self._torch_device).eval()
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self._transform = transforms.small_transform
            self.ready = True
        except Exception:
            self._model = None
            self._transform = None
            self.ready = False

    def estimate(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if not self.ready:
            return None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (DEPTH_SIZE, DEPTH_SIZE), interpolation=cv2.INTER_LINEAR)
        sample = self._transform(small)
        tensor = sample["image"] if isinstance(sample, dict) else sample
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self._torch_device)

        with torch.no_grad():
            prediction = self._model(tensor)
            depth = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(frame.shape[0], frame.shape[1]),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_np = depth.float().cpu().numpy()
        depth_np -= depth_np.min()
        depth_np /= max(depth_np.max(), 1e-6)
        return depth_np

    def depth_at_pixel(self, depth_map: Optional[np.ndarray], u: int, v: int, radius: int = 3) -> Optional[float]:
        if depth_map is None:
            return None
        h, w = depth_map.shape[:2]
        u = int(np.clip(u, 0, w - 1))
        v = int(np.clip(v, 0, h - 1))
        y0 = max(0, v - radius)
        y1 = min(h, v + radius + 1)
        x0 = max(0, u - radius)
        x1 = min(w, u + radius + 1)
        patch = depth_map[y0:y1, x0:x1]
        if patch.size == 0:
            return None
        return relative_depth_to_meters(float(np.median(patch)))

    def depth_at_box(
        self, depth_map: Optional[np.ndarray], x1: int, y1: int, x2: int, y2: int
    ) -> Optional[float]:
        cx = int((x1 + x2) * 0.5)
        cy = int((y1 + y2) * 0.5)
        radius = max(2, int(min(x2 - x1, y2 - y1) * 0.15))
        return self.depth_at_pixel(depth_map, cx, cy, radius)

    def metric_depth_map(self, relative_map: np.ndarray) -> np.ndarray:
        return 1.5 + (1.0 - relative_map) * 14.0
