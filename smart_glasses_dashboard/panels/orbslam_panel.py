import cv2
import numpy as np

from panels.base_panel import PanelWorker
from utils.frame_utils import COLOR_ORB

ORB_FEATURES = 600
MATCH_DRAW_LIMIT = 25
POSE_MATCH_LIMIT = 40


class OrbslamPanel(PanelWorker):
    def __init__(self, input_queue, on_frame=None, panel_width=640, panel_height=480):
        super().__init__(
            "ORB-SLAM",
            COLOR_ORB,
            input_queue,
            on_frame=on_frame,
            panel_width=panel_width,
            panel_height=panel_height,
        )
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
        self._trajectory: list[tuple[float, float]] = [(0.0, 0.0)]
        self._pos = np.array([0.0, 0.0], dtype=np.float64)
        self._trajectory_size = 100
        self._gray = None

    def reset_state(self) -> None:
        self._prev_kp = None
        self._prev_des = None
        self._trajectory = [(0.0, 0.0)]
        self._pos = np.array([0.0, 0.0], dtype=np.float64)

    def process(self, frame: np.ndarray, metadata: dict) -> tuple[np.ndarray, dict]:
        if self._gray is None or self._gray.shape != frame.shape[:2]:
            self._gray = np.empty(frame.shape[:2], dtype=np.uint8)
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, dst=self._gray)
        gray = self._gray

        kp, des = self._orb.detectAndCompute(gray, None)
        keypoint_count = len(kp) if kp else 0
        match_count = 0
        status = "LOST"

        out = frame.copy()
        if kp:
            cv2.drawKeypoints(
                out,
                kp,
                out,
                color=(0, 255, 255),
                flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT,
            )

        if self._prev_des is not None and des is not None and len(des) > 0:
            pair_matches = self._bf.knnMatch(self._prev_des, des, k=2)
            matches = []
            for pair in pair_matches:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    matches.append(m)
            match_count = len(matches)

            if self._prev_kp is not None:
                for m in matches[:MATCH_DRAW_LIMIT]:
                    pt1 = tuple(map(int, self._prev_kp[m.queryIdx].pt))
                    pt2 = tuple(map(int, kp[m.trainIdx].pt))
                    cv2.line(out, pt1, pt2, (0, 200, 255), 1)

            pose_matches = matches[:POSE_MATCH_LIMIT]
            if len(pose_matches) >= 8:
                src_pts = np.float32(
                    [self._prev_kp[m.queryIdx].pt for m in pose_matches]
                ).reshape(-1, 1, 2)
                dst_pts = np.float32([kp[m.trainIdx].pt for m in pose_matches]).reshape(
                    -1, 1, 2
                )
                h, w = gray.shape[:2]
                focal = w
                pp = (w / 2.0, h / 2.0)
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
                    _, _, t, _ = cv2.recoverPose(E, src_pts, dst_pts, focal, pp=pp)
                    self._pos[0] += float(t[0])
                    self._pos[1] += float(t[2])
                    self._trajectory.append((self._pos[0], self._pos[1]))

        if match_count >= 30:
            status = "TRACKING"

        self._prev_kp = kp
        self._prev_des = des

        self._draw_trajectory(out)

        return out, {
            "keypoint_count": keypoint_count,
            "match_count": match_count,
            "status": status,
        }

    def process_disabled(self, frame: np.ndarray, metadata: dict) -> tuple[np.ndarray, dict]:
        return frame, {
            "disabled": True,
            "keypoint_count": 0,
            "match_count": 0,
            "status": "OFF",
        }

    def build_header_lines(self, stats: dict, metadata: dict) -> list[str]:
        if stats.get("disabled"):
            return [
                self._fps_line(),
                "ORB OFF",
                self._source_line(metadata),
            ]
        return [
            self._fps_line(),
            f"KP: {stats.get('keypoint_count', 0)}",
            f"Match: {stats.get('match_count', 0)}",
            stats.get("status", "LOST"),
            self._source_line(metadata),
        ]

    def _draw_trajectory(self, frame: np.ndarray) -> None:
        size = self._trajectory_size
        margin = 8
        x0, y0 = margin, frame.shape[0] - size - margin

        cv2.rectangle(frame, (x0, y0), (x0 + size, y0 + size), (20, 20, 20), -1)
        cv2.rectangle(frame, (x0, y0), (x0 + size, y0 + size), COLOR_ORB, 1)

        if len(self._trajectory) < 2:
            return

        pts = self._trajectory
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-3)
        cx = (max(xs) + min(xs)) / 2.0
        cy = (max(ys) + min(ys)) / 2.0
        half = size * 0.4

        prev = None
        for x, y in pts:
            sx = int(x0 + size / 2 + (x - cx) / span * half)
            sy = int(y0 + size / 2 + (y - cy) / span * half)
            if prev is not None:
                cv2.line(frame, prev, (sx, sy), COLOR_ORB, 1)
            prev = (sx, sy)
