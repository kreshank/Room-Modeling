"""Serve an output directory so viewer/index.html can load ../scene.json."""

from __future__ import annotations

import argparse
import http.server
import socketserver
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", nargs="?", default="../outs/spatial_editor_outputs/my_room")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    if not out_dir.exists():
        raise SystemExit(f"Output directory does not exist: {out_dir}")

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *handler_args, **handler_kwargs):
            super().__init__(*handler_args, directory=str(out_dir), **handler_kwargs)

    with socketserver.TCPServer(("127.0.0.1", args.port), Handler) as httpd:
        print(f"Serving {out_dir}")
        print(f"Open http://127.0.0.1:{args.port}/viewer/index.html")
        httpd.serve_forever()


if __name__ == "__main__":
    raise SystemExit(main())
