"""    python -m contexts.voice        # the voice service, on 127.0.0.1:8091 (the atlas-voice systemd unit runs this)"""

import logging
import os
import threading

import uvicorn

from contexts.gateway.__main__ import load_dotenv
from contexts.voice.config import load_config
from contexts.voice.service import build_session, create_app

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    load_dotenv(os.environ.get("ATLAS_ENV_FILE", ".env"))
    cfg = load_config(os.environ.get("ATLAS_VOICE_CONFIG", "config/voice.yaml"))
    session = build_session(cfg)
    # Bring a paired speaker back after a restart without making startup wait for it
    if cfg.speaker_mac:
        threading.Thread(target=session.speaker.ensure_connected, name="voice-speaker", daemon=True).start()
    uvicorn.run(create_app(session), host="127.0.0.1", port=cfg.port, log_level="warning")
