import logging
import os
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> int:
    """KEY=VALUE lines into the environment, without overriding what is already set. Returns keys loaded.

    Settings read the environment at import time, so this runs before the app is imported."""
    p = Path(path)
    if not p.is_file():
        return 0
    n = 0
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip().removeprefix("export "), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value
            n += 1
    return n


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    loaded = load_dotenv(os.environ.get("ATLAS_ENV_FILE", ".env"))
    logging.getLogger("atlas").info("loaded %d settings from .env", loaded)

    import uvicorn

    from contexts.gateway.app import create_app

    uvicorn.run(create_app(), host=os.environ.get("ATLAS_HOST", "0.0.0.0"), port=int(os.environ.get("ATLAS_PORT", "8000")))
