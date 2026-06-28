"""ORB visual odometry with metric scale from depth."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

ORB_FEATURES = 600
MATCH_DRAW_LIMIT = 25
POSE_MATCH_LIMIT = 40


class PoseTracker:
    """Accumulates camera pose in a world frame (X right, Y down, Z forward)."""

    def __init__(self):
        self._orb = cv2.ORB_create(
            nfeatures=ORB_FEATURES,
            scaleFactor=1.2,
            nlevels=4,
            edgeThreshold=15,
            patchSize=31,
            fastThreshold=12,
        )
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._prev_kp = None
        self._prev_des = None
        self._gray = None
        self.R_wc = np.eye(3, dtype=np.float64)
        self.t_wc = np.zeros(3, dtype=np.float64)
        self.trajectory: list[np.ndarray] = [np.zeros(3, dtype=np.float64)]
        self.yaw_deg = 0.0
        self.distance_m = 0.0
        self.match_count = 0
        self.keypoint_count = 0
        self.status = "LOST"
        self._last_R_rel: Optional[np.ndarray] = None
        self._last_t_rel: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._prev_kp = None
        self._prev_des = None
        self.R_wc = np.eye(3, dtype=np.float64)
        self.t_wc = np.zeros(3, dtype=np.float64)
        self.trajectory = [np.zeros(3, dtype=np.float64)]
        self.yaw_deg = 0.0
        self.distance_m = 0.0
        self.match_count = 0
        self.keypoint_count = 0
        self.status = "LOST"
        self._last_R_rel = None
        self._last_t_rel = None

    def process(
        self,
        frame: np.ndarray,
        depth_map: Optional[np.ndarray],
        depth_estimator,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ) -> tuple[np.ndarray, list]:
        if self._gray is None or self._gray.shape != frame.shape[:2]:
            self._gray = np.empty(frame.shape[:2], dtype=np.uint8)
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, dst=self._gray)
        gray = self._gray

        kp, des = self._orb.detectAndCompute(gray, None)
        self.keypoint_count = len(kp) if kp else 0
        self.match_count = 0
        self.status = "LOST"
        matches_for_draw: list = []
        prev_kp_snapshot = self._prev_kp

        if self._prev_des is not None and des is not None and len(des) > 0 and kp:
            pair_matches = self._bf.knnMatch(self._prev_des, des, k=2)
            matches = []
            for pair in pair_matches:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    matches.append(m)
            self.match_count = len(matches)
            matches_for_draw = matches[:MATCH_DRAW_LIMIT]

            pose_matches = matches[:POSE_MATCH_LIMIT]
            if len(pose_matches) >= 8 and self._prev_kp is not None:
                src_pts = np.float32(
                    [self._prev_kp[m.queryIdx].pt for m in pose_matches]
                ).reshape(-1, 1, 2)
                dst_pts = np.float32([kp[m.trainIdx].pt for m in pose_matches]).reshape(
                    -1, 1, 2
                )
                focal = fx
                pp = (cx, cy)
                E, _ = cv2.findEssentialMat(
                    src_pts,
                    dst_pts,
                    focal,
                    pp=pp,
                    method=cv2.RANSAC,
                    prob=0.999,
                    threshold=1.0,
                )
                if E is not None:
                    _, R_rel, t_rel, _ = cv2.recoverPose(
                        E, src_pts, dst_pts, focal, pp=pp
                    )
                    scale = self._estimate_scale(
                        pose_matches, depth_map, depth_estimator, fx, fy, cx, cy
                    )
                    t_scaled = (t_rel * scale).reshape(3)
                    self._integrate_pose(R_rel, t_scaled)
                    self._last_R_rel = R_rel
                    self._last_t_rel = t_scaled

        if self.match_count >= 30:
            self.status = "TRACKING"

        self._prev_kp = kp
        self._prev_des = des
        return kp if kp else [], matches_for_draw, prev_kp_snapshot

    def _estimate_scale(
        self,
        matches,
        depth_map: Optional[np.ndarray],
        depth_estimator,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ) -> float:
        if depth_map is None or self._prev_kp is None:
            return 0.12

        depths = []
        for m in matches[:20]:
            u, v = self._prev_kp[m.queryIdx].pt
            d = depth_estimator.depth_at_pixel(depth_map, int(u), int(v), radius=4)
            if d is not None and d > 0.3:
                depths.append(d)
        if not depths:
            return 0.12
        return float(np.median(depths)) * 0.04

    def _integrate_pose(self, R_rel: np.ndarray, t_rel: np.ndarray) -> None:
        T_c1_c2 = np.eye(4, dtype=np.float64)
        T_c1_c2[:3, :3] = R_rel.T
        T_c1_c2[:3, 3] = -R_rel.T @ t_rel

        T_w_c = np.eye(4, dtype=np.float64)
        T_w_c[:3, :3] = self.R_wc
        T_w_c[:3, 3] = self.t_wc

        T_w_c_new = T_w_c @ T_c1_c2
        prev_t = self.t_wc.copy()
        self.R_wc = T_w_c_new[:3, :3]
        self.t_wc = T_w_c_new[:3, 3]
        self.trajectory.append(self.t_wc.copy())
        delta = float(np.linalg.norm(self.t_wc - prev_t))
        self.distance_m += delta
        self.yaw_deg = float(np.degrees(np.arctan2(self.R_wc[0, 2], self.R_wc[2, 2])))

    def world_to_camera(self, points_world: np.ndarray) -> np.ndarray:
        R_cw = self.R_wc.T
        t = self.t_wc
        return (R_cw @ (points_world - t).T).T

    def camera_to_world(self, points_cam: np.ndarray) -> np.ndarray:
        return (self.R_wc @ points_cam.T).T + self.t_wc
