"""What this machine is, as far as it can tell by itself: telemetry's guard against writing to the wrong log."""

from pathlib import Path

TEGRA_RELEASE = Path("/etc/nv_tegra_release")  # present on every Jetson Linux (L4T) install


def on_jetson(release_file: Path | None = None) -> bool:
    return (release_file or TEGRA_RELEASE).exists()
