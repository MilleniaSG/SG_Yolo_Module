"""Monocular 3D bounding-box estimation and wireframe rendering for YOLO detections."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

# Real-world priors: (height_m, width_m, depth_m) for common COCO classes.
CLASS_DIMENSIONS: dict[str, tuple[float, float, float]] = {
    "person": (1.72, 0.55, 0.35),
    "bicycle": (1.10, 0.60, 1.80),
    "car": (1.45, 1.80, 4.20),
    "motorcycle": (1.15, 0.80, 2.10),
    "bus": (3.00, 2.55, 10.50),
    "truck": (2.80, 2.40, 7.50),
    "boat": (2.00, 2.50, 6.00),
    "bench": (0.90, 1.50, 0.55),
    "bird": (0.25, 0.30, 0.30),
    "cat": (0.30, 0.35, 0.55),
    "dog": (0.55, 0.45, 0.90),
    "horse": (1.60, 0.70, 2.20),
    "sheep": (0.90, 0.60, 1.10),
    "cow": (1.40, 0.90, 2.00),
    "elephant": (2.80, 2.50, 4.50),
    "bear": (1.20, 0.90, 1.80),
    "zebra": (1.50, 0.70, 2.20),
    "giraffe": (4.50, 1.20, 1.50),
    "backpack": (0.45, 0.35, 0.20),
    "umbrella": (1.20, 1.00, 1.00),
    "handbag": (0.30, 0.35, 0.15),
    "suitcase": (0.55, 0.40, 0.25),
    "bottle": (0.28, 0.08, 0.08),
    "cup": (0.12, 0.09, 0.09),
    "chair": (0.90, 0.50, 0.50),
    "couch": (0.85, 2.00, 0.90),
    "potted plant": (0.55, 0.40, 0.40),
    "bed": (0.55, 2.00, 1.80),
    "dining table": (0.75, 1.40, 0.90),
    "tv": (0.55, 1.10, 0.12),
    "laptop": (0.02, 0.35, 0.25),
    "cell phone": (0.15, 0.08, 0.01),
    "book": (0.25, 0.18, 0.03),
    "clock": (0.30, 0.30, 0.08),
    "vase": (0.35, 0.18, 0.18),
    "default": (1.00, 1.00, 1.00),
}

# Cuboid corner indices: front face 0-3, back face 4-7
_FRONT_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0))
_BACK_EDGES = ((4, 5), (5, 6), (6, 7), (7, 4))
_CONNECT_EDGES = ((0, 4), (1, 5), (2, 6), (3, 7))


def camera_intrinsics(width: int, height: int) -> tuple[float, float, float, float]:
    fx = fy = max(width, height) * 1.15
    cx = width * 0.5
    cy = height * 0.5
    return fx, fy, cx, cy


def class_dimensions(class_name: str) -> tuple[float, float, float]:
    return CLASS_DIMENSIONS.get(class_name.lower(), CLASS_DIMENSIONS["default"])


def unproject_pixel(
    u: float,
    v: float,
    depth_m: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    x = (u - cx) * depth_m / fx
    y = (v - cy) * depth_m / fy
    z = depth_m
    return np.array([x, y, z], dtype=np.float32)


def estimate_3d_box_camera_corners(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    class_name: str,
    frame_shape: tuple[int, int],
    depth_override: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return 8x3 camera-frame corners, 8x2 image corners, depth meters."""
    h_img, w_img = frame_shape
    fx, fy, cx, cy = camera_intrinsics(w_img, h_img)

    real_h, real_w, real_d = class_dimensions(class_name)
    pixel_w = max(x2 - x1, 2.0)
    pixel_h = max(y2 - y1, 2.0)

    if depth_override is not None and depth_override > 0:
        depth = depth_override
        real_w = pixel_w * depth / fx
        real_h = pixel_h * depth / fy
        real_d = max(real_w * 0.55, 0.15)
    else:
        depth = fy * real_h / pixel_h
        measured_w = pixel_w * depth / fx
        real_w = 0.65 * real_w + 0.35 * measured_w

    u_center = (x1 + x2) * 0.5
    v_bottom = y2
    x_cam = (u_center - cx) * depth / fx
    y_bottom = (v_bottom - cy) * depth / fy
    y_center = y_bottom - real_h * 0.5
    hw, hh = real_w * 0.5, real_h * 0.5

    local_corners = np.array(
        [
            [-hw, -hh, 0.0],
            [hw, -hh, 0.0],
            [hw, hh, 0.0],
            [-hw, hh, 0.0],
            [-hw, -hh, real_d],
            [hw, -hh, real_d],
            [hw, hh, real_d],
            [-hw, hh, real_d],
        ],
        dtype=np.float32,
    )
    cam_corners = local_corners + np.array([x_cam, y_center, depth], dtype=np.float32)
    u = fx * cam_corners[:, 0] / cam_corners[:, 2] + cx
    v = fy * cam_corners[:, 1] / cam_corners[:, 2] + cy
    corners_2d = np.stack([u, v], axis=1)
    return cam_corners, corners_2d, float(depth)


def estimate_3d_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    class_name: str,
    frame_shape: tuple[int, int],
    depth_override: Optional[float] = None,
) -> tuple[np.ndarray, float]:
    """Return 8 projected 2D corners and estimated depth (meters)."""
    _, corners_2d, depth = estimate_3d_box_camera_corners(
        x1, y1, x2, y2, class_name, frame_shape, depth_override
    )
    return corners_2d, depth


def draw_3d_cuboid(
    frame: np.ndarray,
    corners_2d: np.ndarray,
    color: tuple[int, int, int],
    label: str = "",
    thickness: int = 2,
) -> None:
    pts = [tuple(map(int, p)) for p in corners_2d]

    back_color = tuple(int(c * 0.55) for c in color)
    for i, j in _BACK_EDGES:
        cv2.line(frame, pts[i], pts[j], back_color, max(1, thickness - 1), cv2.LINE_AA)
    for i, j in _CONNECT_EDGES:
        cv2.line(frame, pts[i], pts[j], color, thickness, cv2.LINE_AA)
    for i, j in _FRONT_EDGES:
        cv2.line(frame, pts[i], pts[j], color, thickness + 1, cv2.LINE_AA)

    # Ground contact hint on front bottom edge
    cv2.line(frame, pts[2], pts[3], (255, 255, 255), 1, cv2.LINE_AA)

    if label:
        anchor = pts[0]
        cv2.putText(
            frame,
            label,
            (anchor[0], max(anchor[1] - 6, 14)),
            cv2.FONT_HERSHEY_DUPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_depth_axis(frame: np.ndarray) -> None:
    """Small 3D axis gizmo in the top-left of the panel."""
    origin = (28, 78)
    cv2.arrowedLine(frame, origin, (origin[0] + 26, origin[1]), (0, 80, 255), 2, tipLength=0.25)
    cv2.arrowedLine(frame, origin, (origin[0], origin[1] + 26), (0, 200, 0), 2, tipLength=0.25)
    cv2.arrowedLine(frame, origin, (origin[0] - 18, origin[1] - 14), (255, 140, 0), 2, tipLength=0.25)
    cv2.putText(frame, "X", (origin[0] + 28, origin[1] + 4), cv2.FONT_HERSHEY_DUPLEX, 0.35, (200, 200, 200), 1)
    cv2.putText(frame, "Y", (origin[0] - 4, origin[1] + 38), cv2.FONT_HERSHEY_DUPLEX, 0.35, (200, 200, 200), 1)
    cv2.putText(frame, "Z", (origin[0] - 30, origin[1] - 12), cv2.FONT_HERSHEY_DUPLEX, 0.35, (200, 200, 200), 1)
