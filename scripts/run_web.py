from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import app, cfg


if __name__ == "__main__":
    app.run(host=cfg["flask"]["host"], port=cfg["flask"]["port"], debug=cfg["flask"]["debug"])
