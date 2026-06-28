import cv2
import numpy as np

BG_COLOR = (13, 13, 13)
BORDER_COLOR = (30, 30, 46)
HEADER_HEIGHT = 36

COLOR_ORIGINAL = (255, 212, 0)   # #00d4ff BGR
COLOR_YOLO = (0, 255, 170)       # #aaff00 BGR
COLOR_ORB = (0, 107, 255)        # #ff6b00 BGR

MONO_FONT = cv2.FONT_HERSHEY_DUPLEX
MONO_SCALE = 0.45
MONO_THICKNESS = 1


def sample_pixel_bgr(frame: np.ndarray, u: float, v: float) -> list[int]:
    """Sample BGR color from a video frame at pixel (u, v)."""
    h, w = frame.shape[:2]
    ui = int(np.clip(u, 0, w - 1))
    vi = int(np.clip(v, 0, h - 1))
    b, g, r = frame[vi, ui]
    return [int(b), int(g), int(r)]


def draw_header_bar_inplace(
    frame: np.ndarray,
    title: str,
    accent_color: tuple[int, int, int],
    lines: list[str],
) -> np.ndarray:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, HEADER_HEIGHT), accent_color, -1)

    cv2.putText(
        frame,
        title,
        (8, 24),
        MONO_FONT,
        MONO_SCALE,
        (255, 255, 255),
        MONO_THICKNESS,
        cv2.LINE_AA,
    )

    if lines:
        stats = "  |  ".join(lines)
        (text_w, _), _ = cv2.getTextSize(stats, MONO_FONT, MONO_SCALE, MONO_THICKNESS)
        x = max(8, w - text_w - 8)
        cv2.putText(
            frame,
            stats,
            (x, 24),
            MONO_FONT,
            MONO_SCALE,
            (255, 255, 255),
            MONO_THICKNESS,
            cv2.LINE_AA,
        )

    return frame


def class_color(class_id: int) -> tuple[int, int, int]:
    hue = int((class_id * 47) % 180)
    hsv = np.uint8([[[hue, 200, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_header_bar(
    frame: np.ndarray,
    title: str,
    accent_color: tuple[int, int, int],
    lines: list[str],
) -> np.ndarray:
    out = frame.copy()
    return draw_header_bar_inplace(out, title, accent_color, lines)


def make_placeholder(
    width: int,
    height: int,
    panel_name: str,
    accent_color: tuple[int, int, int],
) -> np.ndarray:
    frame = np.full((height, width, 3), BG_COLOR, dtype=np.uint8)
    cv2.rectangle(frame, (0, 0), (width - 1, height - 1), BORDER_COLOR, 2)
    frame = draw_header_bar(frame, panel_name, accent_color, ["waiting for source..."])

    text = panel_name
    (text_w, text_h), _ = cv2.getTextSize(text, MONO_FONT, 0.8, 2)
    x = (width - text_w) // 2
    y = (height + text_h) // 2
    cv2.putText(
        frame,
        text,
        (x, y),
        MONO_FONT,
        0.8,
        accent_color,
        2,
        cv2.LINE_AA,
    )
    return frame


def format_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def frame_to_jpeg_bytes(frame: np.ndarray, quality: int = 80) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Failed to encode frame to JPEG")
    return buf.tobytes()
