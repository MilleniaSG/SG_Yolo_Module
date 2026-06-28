import platform
import re
import subprocess

import cv2


def _parse_macos_camera_names() -> list[str]:
    try:
        result = subprocess.run(
            ["system_profiler", "SPCameraDataType"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    names: list[str] = []
    for line in result.stdout.splitlines():
        match = re.match(r"^    ([^ ].+):$", line)
        if not match:
            continue
        name = match.group(1).strip()
        if name.lower() == "camera":
            continue
        names.append(name)
    return names


def _probe_camera_indices(max_index: int = 10) -> list[int]:
    available: list[int] = []
    for index in range(max_index + 1):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            available.append(index)
            cap.release()
    return available


def enumerate_cameras() -> list[dict]:
    """Return list of dicts: {index, name, is_iphone}."""
    indices = _probe_camera_indices()
    cameras: list[dict] = []

    if platform.system() == "Darwin":
        profiler_names = _parse_macos_camera_names()
        for i, index in enumerate(indices):
            if i < len(profiler_names):
                name = profiler_names[i]
            else:
                name = f"Camera {index}"
            cameras.append(
                {
                    "index": index,
                    "name": name,
                    "is_iphone": "iphone" in name.lower(),
                }
            )
    else:
        for index in indices:
            cameras.append(
                {
                    "index": index,
                    "name": f"Camera {index}",
                    "is_iphone": False,
                }
            )

    return cameras


from typing import Optional


def get_default_iphone_index() -> Optional[int]:
    for camera in enumerate_cameras():
        if camera["is_iphone"]:
            return camera["index"]
    return None
