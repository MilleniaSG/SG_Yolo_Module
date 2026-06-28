"""FastAPI web server for the Smart Glasses CV Dashboard."""

from __future__ import annotations

import asyncio
import os
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from source_manager import SPEED_OPTIONS
from web.dashboard import dashboard

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
UPLOAD_DIR = APP_DIR.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Smart Glasses CV Dashboard")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
async def startup() -> None:
    dashboard.start()


@app.on_event("shutdown")
async def shutdown() -> None:
    dashboard.stop()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/cameras")
async def api_cameras() -> dict:
    cameras = dashboard.get_cameras()
    return {"cameras": cameras, "iphone_index": next(
        (c["index"] for c in cameras if c.get("is_iphone")), None
    )}


@app.post("/api/source/webcam")
async def api_webcam(payload: dict) -> dict:
    index = int(payload.get("index", 0))
    dashboard.switch_webcam(index)
    return {"ok": True}


@app.post("/api/source/continuity")
async def api_continuity(payload: dict) -> dict:
    index = payload.get("index")
    dashboard.switch_continuity(int(index) if index is not None else None)
    return {"ok": True}


@app.post("/api/source/file")
async def api_upload(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    dashboard.switch_file(str(dest))
    return {"ok": True, "path": str(dest)}


@app.post("/api/playback/toggle")
async def api_toggle() -> dict:
    dashboard.toggle_play()
    return {"ok": True}


@app.post("/api/playback/speed")
async def api_speed(payload: dict) -> dict:
    index = int(payload.get("index", 3))
    dashboard.set_speed_index(index)
    return {"ok": True, "speed": SPEED_OPTIONS[index]}


@app.post("/api/playback/loop")
async def api_loop(payload: dict) -> dict:
    dashboard.set_loop(bool(payload.get("enabled", False)))
    return {"ok": True}


@app.post("/api/playback/seek")
async def api_seek(payload: dict) -> dict:
    dashboard.seek(float(payload.get("fraction", 0.0)))
    return {"ok": True}


@app.post("/api/toggle/yolo")
async def api_toggle_yolo() -> dict:
    enabled = dashboard.toggle_yolo()
    return {"ok": True, "enabled": enabled}


@app.post("/api/toggle/orb")
async def api_toggle_orb() -> dict:
    enabled = dashboard.toggle_orb()
    return {"ok": True, "enabled": enabled}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    try:
        while True:
            await ws.send_json(dashboard.get_state())
            await asyncio.sleep(0.066)  # ~15 fps to browser
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
