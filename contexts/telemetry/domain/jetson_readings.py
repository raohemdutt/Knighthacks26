"""What the Jetson's own tools report: a tegrastats line as rail milliwatts and temperatures, `nvpmodel -q` as a mode,
and the INA3221 power monitor's averaging settings.

    line = parse_tegrastats_line(text)   # None when the text is not a tegrastats statistics line
    line.rails_mw                        # {"VDD_IN": 7732.0, "VDD_CPU_GPU_CV": 2120.0, "VDD_SOC": 2524.0}
    line.temps_c                         # {"cpu": 51.562, "soc2": 51.093, "gpu": 50.875, ...}
    parse_nvpmodel(output)               # "MAXN_SUPER"; None when the output names no mode

tegrastats prints one line per interval. A power rail appears as `NAME current/average`, both in mW: on
JetPack 6 with the unit on each number (`VDD_IN 7732mW/6981mW`), in the developer guide's own example without
it (`VDDRQ 312/234`). Only the first number is a reading. The second is tegrastats' running average since it
started: in the recorded Orin Nano lines it stays near 7000 mW while the current value jumps from 7732 to
19955 mW, so integrating it would smear every request over the whole run. A temperature appears as
`name@degreesC` (`gpu@50.875C`, `soc2@48C`).

Sources: NVIDIA Jetson Linux Developer Guide r36.4, "Tegrastats Utility" (field formats and units) and
"Platform Power and Performance" (`nvpmodel -q`); recorded output in fixtures/telemetry/jetson_*.txt.
"""

import re
from dataclasses import dataclass

# NAME current/average, the unit on both numbers or on neither; never followed by more unit letters (RAM 1603/7620MB)
_RAIL = re.compile(r"(?<![\w@])([A-Z][A-Z0-9_]*) (\d+(?:\.\d+)?)(mW)?/(\d+(?:\.\d+)?)(mW)?(?=\s|$)")
_TEMP = re.compile(r"(?<![\w@])([A-Za-z]\w*)@(-?\d+(?:\.\d+)?)C(?=\s|$)")
_RAM = re.compile(r"(?<!\w)RAM \d+/\d+MB")
_MODE = re.compile(r"NV Power Mode:\s*(\S+)")

LOWEST_REAL_C = -100.0  # colder than any running Jetson reports: a value at or below it is a placeholder, not a reading


@dataclass(frozen=True)
class Ina3221Settings:
    """How the INA3221 power monitor averages: its readings are only as fresh as `update_ms`."""

    samples: int | None  # conversions averaged into each reading (1, 4, 16 ... 1024)
    update_ms: float | None  # ms between new readings; polling faster re-reads the same value
    hwmon: str | None = None  # the sysfs directory they were read from, when one was found


@dataclass(frozen=True)
class TegrastatsLine:
    rails_mw: dict[str, float]  # rail -> its current power in mW (the first number), in the order printed
    temps_c: dict[str, float]  # sensor -> degrees C


def parse_tegrastats_line(text: str) -> TegrastatsLine | None:
    """The rails and temperatures in one tegrastats line; None when the text holds no tegrastats statistics.

    A statistics line without power rails (tegrastats run where it cannot read the power monitor) comes back
    with empty `rails_mw`; the caller decides what a sample without power is worth."""
    rails: dict[str, float] = {}
    for m in _RAIL.finditer(text):
        name, current, unit, _average, average_unit = m.groups()
        if (unit is None) != (average_unit is None):
            continue  # one number with a unit and one without is not a rail reading
        if unit is None and not name.startswith("VDD"):
            continue  # without units, only a VDD rail is taken to be power
        rails.setdefault(name, float(current))
    temps: dict[str, float] = {}
    for m in _TEMP.finditer(text):
        celsius = float(m.group(2))
        if celsius > LOWEST_REAL_C:
            temps.setdefault(m.group(1), celsius)
    if not rails and not temps and not _RAM.search(text):
        return None
    return TegrastatsLine(rails_mw=rails, temps_c=temps)


def parse_nvpmodel(output: str) -> str | None:
    """The power mode name in `nvpmodel -q` output (plain or --verbose), e.g. "MAXN_SUPER" or "15W"."""
    m = _MODE.search(output)
    return m.group(1) if m else None
