"""What the Jetson's kernel says about its power sensor, its overcurrent events and its CPU time (read-only).

    sysfs = JetsonSysfs()            # root="/" on the box; tests point it at a directory tree of their own
    sysfs.ina3221()                  # Ina3221Settings(samples=..., update_ms=..., hwmon=...)
    sysfs.oc_events()                # {"oc1": 0, "oc2": 0, "oc3": 194}, cumulative since boot
    sysfs.busy_cpu_s()               # CPU seconds all cores spent busy since boot (/proc/stat)
    sysfs.process_cpu_s(pid)         # CPU seconds one process has used (/proc/<pid>/stat)

- The INA3221 power monitor (I2C address 0x40 on the Orin Nano) lives under
  /sys/bus/i2c/drivers/ina3221/<bus-addr>/hwmon/hwmon<x>/ (NVIDIA Jetson Linux Developer Guide r36.4,
  "Platform Power and Performance"). Its `samples` is how many conversions are averaged into each reading and
  `update_interval` the time in ms between new readings, update_interval = channels x samples x (bus + shunt
  conversion time) (Linux kernel documentation, "Kernel driver ina3221").
- The overcurrent event counters are /sys/class/hwmon/hwmon<x>/oc<n>_event_cnt (same NVIDIA page, "Overcurrent
  Event Status"; recorded output in fixtures/telemetry/jetson_oc_events.txt).

A file that is missing or unreadable gives None, or no entry, never a made-up value. Paths are found once and
kept until `forget()`, so a sample at 100 ms costs a few small reads and no directory scans.
"""

import logging
import os
from pathlib import Path

from contexts.telemetry.domain.jetson_readings import Ina3221Settings

log = logging.getLogger("atlas.telemetry")

INA3221_HWMON = "sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*"
HWMON = "sys/class/hwmon/hwmon*"
OC_COUNTER_SUFFIX = "_event_cnt"


class JetsonSysfs:
    def __init__(self, root: str | Path = "/") -> None:
        self.root = Path(root)
        self._ina: Path | None = None
        self._ina_found = False
        self._oc: list[tuple[str, Path]] | None = None
        self._unreadable: set[Path] = set()

    def forget(self) -> None:
        """Find the sensor and counter files again on the next read (a sampler does this on every start)."""
        self._ina, self._ina_found, self._oc = None, False, None

    # --- the INA3221 power monitor ---

    def ina3221(self) -> Ina3221Settings:
        hwmon = self._ina3221_dir()
        if hwmon is None:
            return Ina3221Settings(samples=None, update_ms=None)
        samples = self._read_int(hwmon / "samples")
        update = self._read_int(hwmon / "update_interval")
        return Ina3221Settings(samples=samples, update_ms=None if update is None else float(update), hwmon=str(hwmon))

    def _ina3221_dir(self) -> Path | None:
        if not self._ina_found:
            self._ina, self._ina_found = self._find_ina3221(), True
            if self._ina is None:
                log.warning("no INA3221 hwmon directory under %s; samples will not record its settings", self.root)
        return self._ina

    def _find_ina3221(self) -> Path | None:
        candidates = sorted(self.root.glob(INA3221_HWMON))
        if not candidates:  # the driver's directory may be missing; hwmon names the chip either way
            candidates = [d for d in sorted(self.root.glob(HWMON)) if self._read_text(d / "name") == "ina3221"]
        for d in candidates:  # boards with two monitors: the one that measures the module input
            if any(self._read_text(label) == "VDD_IN" for label in sorted(d.glob("in*_label"))):
                return d
        return candidates[0] if candidates else None

    # --- overcurrent events ---

    def oc_events(self) -> dict[str, int]:
        if self._oc is None:
            self._oc = []
            for path in sorted(self.root.glob(f"{HWMON}/oc*{OC_COUNTER_SUFFIX}")):
                name = path.name[: -len(OC_COUNTER_SUFFIX)]
                if name not in {n for n, _ in self._oc}:
                    self._oc.append((name, path))
        counts = {}
        for name, path in self._oc:
            value = self._read_int(path)
            if value is not None:
                counts[name] = value
        return counts

    # --- CPU time, for measuring what the sampler costs ---

    def busy_cpu_s(self) -> float:
        """CPU seconds all cores together spent busy (not idle, not waiting on I/O) since boot."""
        return busy_cpu_s_of((self.root / "proc" / "stat").read_text(), clock_ticks())

    def process_cpu_s(self, pid: int) -> float:
        """CPU seconds (user + system) process `pid` has used so far."""
        return process_cpu_s_of((self.root / "proc" / str(pid) / "stat").read_text(), clock_ticks())

    # --- reading one value ---

    def _read_text(self, path: Path) -> str | None:
        try:
            return path.read_text().strip()
        except OSError as exc:
            if path not in self._unreadable:
                self._unreadable.add(path)
                log.warning("cannot read %s: %s", path, exc)
            return None

    def _read_int(self, path: Path) -> int | None:
        text = self._read_text(path)
        try:
            return None if text is None else int(text)
        except ValueError:
            log.warning("%s holds %r, not a whole number", path, text)
            return None


def clock_ticks() -> int:
    """Kernel clock ticks per second (USER_HZ), the unit of /proc CPU times."""
    return os.sysconf("SC_CLK_TCK")


def busy_cpu_s_of(proc_stat: str, ticks: int) -> float:
    """Busy CPU seconds from /proc/stat's aggregate `cpu` line: user + nice + system + irq + softirq.

    The columns are user nice system idle iowait irq softirq steal guest guest_nice, in USER_HZ ticks (Linux
    kernel documentation, "The /proc Filesystem"); idle, iowait and steal are not this machine running code,
    and the guest columns are left out."""
    for line in proc_stat.splitlines():
        fields = line.split()
        if fields and fields[0] == "cpu":
            user, nice, system, _idle, _iowait, irq, softirq = (int(v) for v in fields[1:8])
            return (user + nice + system + irq + softirq) / ticks
    raise ValueError("no aggregate cpu line in /proc/stat")


def process_cpu_s_of(pid_stat: str, ticks: int) -> float:
    """utime + stime (fields 14 and 15 of /proc/<pid>/stat), in seconds. The command name in field 2 may hold
    spaces or parentheses, so fields are counted from the last ')'."""
    after = pid_stat[pid_stat.rindex(")") + 2:].split()
    utime, stime = int(after[11]), int(after[12])  # fields 14 and 15 overall; `after` starts at field 3
    return (utime + stime) / ticks
