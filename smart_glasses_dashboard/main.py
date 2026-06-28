"""Launch the Smart Glasses CV Dashboard web server."""

import uvicorn


def main() -> None:
    uvicorn.run(
        "web.server:app",
        host="127.0.0.1",
        port=8765,
        reload=False,
        log_level="info",
        ws_max_size=16 * 1024 * 1024,
    )


if __name__ == "__main__":
    main()
