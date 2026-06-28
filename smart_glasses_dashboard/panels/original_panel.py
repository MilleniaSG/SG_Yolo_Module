import numpy as np

from panels.base_panel import PanelWorker


class OriginalPanel(PanelWorker):
    def __init__(self, input_queue, on_frame=None, panel_width=640, panel_height=480):
        from utils.frame_utils import COLOR_ORIGINAL

        super().__init__(
            "ORIGINAL FEED",
            COLOR_ORIGINAL,
            input_queue,
            on_frame=on_frame,
            panel_width=panel_width,
            panel_height=panel_height,
        )

    def process(self, frame: np.ndarray, metadata: dict) -> tuple[np.ndarray, dict]:
        return frame, {
            "resolution": metadata.get("resolution", ""),
            "frame_number": metadata.get("frame_number", 0),
        }

    def build_header_lines(self, stats: dict, metadata: dict) -> list[str]:
        return [
            self._fps_line(),
            f"Res: {stats.get('resolution', metadata.get('resolution', ''))}",
            f"Frame: {stats.get('frame_number', metadata.get('frame_number', 0))}",
            self._source_line(metadata),
        ]
