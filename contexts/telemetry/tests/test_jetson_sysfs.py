"""Reading the INA3221 settings, the overcurrent counters and CPU time from a sysfs and procfs tree laid out like the
Jetson's. The overcurrent files come from recorded output (fixtures/telemetry/jetson_oc_events.txt); the INA3221
values are made up, since no recorded value for the Orin Nano exists yet."""

from pathlib import Path

import pytest

from contexts.telemetry.domain.jetson_readings import Ina3221Settings
from contexts.telemetry.infrastructure import jetson_sysfs
from contexts.telemetry.infrastructure.jetson_sysfs import JetsonSysfs, busy_cpu_s_of, process_cpu_s_of

FIXTURES = Path(__file__).resolve().parents[3] / "fixtures" / "telemetry"
INA = "sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1"


def write(root: Path, relative: str, text: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")
    return path


def ina3221(root: Path, where: str = INA, samples: str = "4", update: str = "26", labels=("VDD_IN", "VDD_CPU_GPU_CV", "VDD_SOC")):
    for n, label in enumerate(labels, 1):
        write(root, f"{where}/in{n}_label", label)
    write(root, f"{where}/samples", samples)
    write(root, f"{where}/update_interval", update)


def recorded_oc_files(root: Path) -> None:
    """The overcurrent files exactly as `grep "" /sys/class/hwmon/hwmon*/oc*` printed them on an Orin Nano."""
    path = FIXTURES / "jetson_oc_events.txt"
    assert path.exists(), f"expected the recorded fixture {path}"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("/sys/"):
            file, value = line.rsplit(":", 1)
            write(root, file.lstrip("/"), value)


def test_the_ina3221_settings_come_from_its_hwmon_directory_under_the_i2c_driver(tmp_path):
    ina3221(tmp_path)
    assert JetsonSysfs(tmp_path).ina3221() == Ina3221Settings(samples=4, update_ms=26.0, hwmon=str(tmp_path / INA))


def test_with_two_power_monitors_the_one_measuring_module_input_is_read(tmp_path):
    other = "sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon0"  # sorts first, measures other rails
    ina3221(tmp_path, where=other, samples="1", update="6", labels=("VDD_GPU_SOC", "VDD_CPU_CV", "VIN_SYS_5V0"))
    ina3221(tmp_path, where="sys/bus/i2c/drivers/ina3221/1-0041/hwmon/hwmon5", samples="16", update="105")
    assert JetsonSysfs(tmp_path).ina3221() == Ina3221Settings(
        samples=16, update_ms=105.0, hwmon=str(tmp_path / "sys/bus/i2c/drivers/ina3221/1-0041/hwmon/hwmon5"))


def test_without_the_i2c_driver_directory_the_hwmon_named_ina3221_is_read(tmp_path):
    write(tmp_path, "sys/class/hwmon/hwmon0/name", "soctherm_oc")
    write(tmp_path, "sys/class/hwmon/hwmon2/name", "ina3221")
    ina3221(tmp_path, where="sys/class/hwmon/hwmon2", samples="64", update="420")
    assert JetsonSysfs(tmp_path).ina3221() == Ina3221Settings(
        samples=64, update_ms=420.0, hwmon=str(tmp_path / "sys/class/hwmon/hwmon2"))


def test_no_power_monitor_means_no_settings_not_made_up_ones(tmp_path):
    assert JetsonSysfs(tmp_path).ina3221() == Ina3221Settings(samples=None, update_ms=None, hwmon=None)


def test_a_settings_file_that_is_missing_or_not_a_number_reads_as_none(tmp_path):
    ina3221(tmp_path, samples="four")
    (tmp_path / INA / "update_interval").unlink()
    assert JetsonSysfs(tmp_path).ina3221() == Ina3221Settings(samples=None, update_ms=None, hwmon=str(tmp_path / INA))


def test_the_overcurrent_counters_read_as_recorded_and_the_throttle_switches_are_not_counters(tmp_path):
    recorded_oc_files(tmp_path)
    assert JetsonSysfs(tmp_path).oc_events() == {"oc1": 0, "oc2": 0, "oc3": 194}


def test_each_read_gives_the_counters_current_values(tmp_path):
    recorded_oc_files(tmp_path)
    sysfs = JetsonSysfs(tmp_path)
    assert sysfs.oc_events()["oc3"] == 194
    write(tmp_path, "sys/class/hwmon/hwmon3/oc3_event_cnt", "195")
    assert sysfs.oc_events() == {"oc1": 0, "oc2": 0, "oc3": 195}


def test_counter_files_are_found_once_and_again_only_after_forget(tmp_path):
    recorded_oc_files(tmp_path)
    sysfs = JetsonSysfs(tmp_path)
    assert sysfs.oc_events() == {"oc1": 0, "oc2": 0, "oc3": 194}
    write(tmp_path, "sys/class/hwmon/hwmon3/oc4_event_cnt", "7")
    assert "oc4" not in sysfs.oc_events()
    sysfs.forget()
    assert sysfs.oc_events() == {"oc1": 0, "oc2": 0, "oc3": 194, "oc4": 7}


def test_a_counter_that_cannot_be_read_is_left_out(tmp_path):
    recorded_oc_files(tmp_path)
    sysfs = JetsonSysfs(tmp_path)
    sysfs.oc_events()
    (tmp_path / "sys/class/hwmon/hwmon3/oc2_event_cnt").unlink()
    assert sysfs.oc_events() == {"oc1": 0, "oc3": 194}


def test_busy_cpu_time_counts_user_nice_system_irq_and_softirq_only():
    stat = ("cpu  1000 50 400 90000 300 20 30 7 11 13\n"
            "cpu0 500 25 200 45000 150 10 15 3 5 6\n"
            "intr 123456\n")
    assert busy_cpu_s_of(stat, ticks=100) == pytest.approx((1000 + 50 + 400 + 20 + 30) / 100)


def test_proc_stat_without_the_aggregate_line_is_an_error():
    with pytest.raises(ValueError, match="no aggregate cpu line"):
        busy_cpu_s_of("cpu0 1 2 3 4 5 6 7\n", ticks=100)


def test_process_cpu_time_is_utime_plus_stime_even_when_the_name_has_spaces_and_parentheses():
    # fields: pid (comm) state ppid pgrp session tty_nr tpgid flags minflt cminflt majflt cmajflt utime stime ...
    stat = "4242 (tegra (stats) x) S 1 4242 4242 0 -1 4194560 120 0 0 0 250 75 0 0 20 0 1 0 900 1000 200"
    assert process_cpu_s_of(stat, ticks=100) == pytest.approx(3.25)


def test_cpu_time_reads_from_the_proc_tree(tmp_path, monkeypatch):
    write(tmp_path, "proc/stat", "cpu  200 0 100 5000 0 0 0 0 0 0")
    write(tmp_path, "proc/77/stat", "77 (tegrastats) S 1 77 77 0 -1 0 0 0 0 0 30 20 0 0 20 0 1 0 1 1 1")
    monkeypatch.setattr(jetson_sysfs, "clock_ticks", lambda: 100)  # os.sysconf exists only on Linux
    sysfs = JetsonSysfs(tmp_path)
    assert (sysfs.busy_cpu_s(), sysfs.process_cpu_s(77)) == (pytest.approx(3.0), pytest.approx(0.5))
