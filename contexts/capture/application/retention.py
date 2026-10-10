"""Keep saved pictures for the number of days the Vault policy says (`keep_days`; None = forever).

    removed = prune_frames("data", keep_days=7, now=datetime.now(UTC))

Only data/frames/<camera>/<ts_ms>.jpg are considered, by the time in the file name (the frame_ref time), so a
picture's age does not depend on file dates. Links are never followed or removed, and nothing outside data/frames
is touched. What Memory wrote down about a picture (captions, episodes) stays: the phone shows "frame not stored".
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

log = logging.getLogger("atlas.capture")


def prune_frames(data_dir: str | Path, keep_days: int | None, now: datetime) -> int:
    if not keep_days:
        return 0
    cutoff_ms = int((now - timedelta(days=keep_days)).timestamp() * 1000)
    removed = 0
    for path in (Path(data_dir) / "frames").glob("*/*.jpg"):
        if path.is_symlink() or not path.stem.isdigit() or int(path.stem) >= cutoff_ms:
            continue
        try:
            path.unlink()
            removed += 1
        except OSError:
            log.warning("could not remove %s", path)
    if removed:
        log.info("retention: removed %d pictures older than %d days", removed, keep_days)
    return removed
