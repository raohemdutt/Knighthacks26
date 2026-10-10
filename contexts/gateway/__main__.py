import logging
import os

import uvicorn

from contexts.gateway.app import create_app

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    uvicorn.run(create_app(), host=os.environ.get("ATLAS_HOST", "0.0.0.0"), port=int(os.environ.get("ATLAS_PORT", "8000")))
