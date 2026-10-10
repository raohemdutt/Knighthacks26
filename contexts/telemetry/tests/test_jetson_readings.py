"""Reading tegrastats and nvpmodel output, on lines recorded from Orin Nano developer kits (fixtures/telemetry)."""

from pathlib import Path

import pytest

from contexts.telemetry.domain.jetson_readings import TegrastatsLine, parse_nvpmodel, parse_tegrastats_line

FIXTURES = Path(__file__).resolve().parents[3] / "fixtures" / "telemetry"


def recorded(name: str) -> list[str]:
    path = FIXTURES / name
    assert path.exists(), f"expected the recorded fixture {path}"
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]


def recorded_text(name: str) -> str:
    return "\n".join(recorded(name)) + "\n"


TEGRASTATS = recorded("jetson_tegrastats_orin_nano.txt")


def test_every_recorded_line_gives_the_current_value_of_each_rail_never_the_running_average():
    parsed = [parse_tegrastats_line(line) for line in TEGRASTATS]
    assert all(isinstance(p, TegrastatsLine) for p in parsed)
    assert [p.rails_mw["VDD_IN"] for p in parsed] == [
        7732, 17730, 19955, 19955,  # an LLM request starting: the average beside these stays near 7000
        19844, 20080, 20040, 20080, 19844,  # heavy GPU load, two cores online
        4629, 4629, 5256, 5818, 5889, 6009, 6009, 5849, 5969, 4903, 4629, 4629, 4629,  # no sudo, a short CPU burst
    ]
    assert [list(p.rails_mw) for p in parsed] == [["VDD_IN", "VDD_CPU_GPU_CV", "VDD_SOC"]] * len(TEGRASTATS)


def test_a_load_step_line_reads_in_full():
    line = parse_tegrastats_line(TEGRASTATS[1])  # 01-20-2025 10:20:46, VDD_IN 17730mW/7006mW
    assert line == TegrastatsLine(
        rails_mw={"VDD_IN": 17730.0, "VDD_CPU_GPU_CV": 9352.0, "VDD_SOC": 3711.0},
        temps_c={"cpu": 52.031, "soc2": 52.031, "soc0": 50.5, "gpu": 54.437, "tj": 54.437, "soc1": 51.125})


def test_a_line_from_tegrastats_run_without_sudo_still_has_every_rail_and_temperature():
    line = parse_tegrastats_line(TEGRASTATS[17])  # 04-24-2025 09:49:20: no EMC_FREQ, GR3D_FREQ without a clock
    assert line == TegrastatsLine(
        rails_mw={"VDD_IN": 5969.0, "VDD_CPU_GPU_CV": 1440.0, "VDD_SOC": 1522.0},
        temps_c={"cpu": 49.187, "soc2": 48.0, "soc0": 48.156, "gpu": 48.781, "tj": 50.125, "soc1": 50.125})


def test_memory_cpu_and_engine_fields_are_never_taken_for_power_rails():
    line = "01-01-2025 00:00:00 RAM 1603/7620MB (lfb 9x4MB) SWAP 0/16384MB (cached 0MB) IRAM 0/252kB(lfb 252kB) " \
           "CPU [0%@960,off] EMC_FREQ 0%@3199 GR3D_FREQ 0%@[306] NVDEC off APE 200 MTS fg 35% bg 65%"
    assert parse_tegrastats_line(line) == TegrastatsLine(rails_mw={}, temps_c={})


def test_the_developer_guide_example_format_without_units_reads_too():
    # Tegrastats Utility, r36.4: "VDDRQ 312/234" (current/average mW) and "MCPU@45C"
    line = "RAM 220/38955MB (lfb 847x4MB) SWAP 0/1978MB (cached 0MB) CPU [1%@102,5%@102,0%@102,0%@102] " \
           "MCPU@45C VDDRQ 312/234 MTS fg 35% bg 65% APE 150"
    assert parse_tegrastats_line(line) == TegrastatsLine(rails_mw={"VDDRQ": 312.0}, temps_c={"MCPU": 45.0})


def test_numbers_without_units_count_as_power_only_on_a_vdd_rail():
    assert parse_tegrastats_line("RAM 1/2MB FOO 312/234 VDD_X 5mW/6") == TegrastatsLine(rails_mw={}, temps_c={})


def test_a_temperature_far_below_anything_a_running_board_reports_is_dropped_as_a_placeholder():
    line = parse_tegrastats_line("RAM 1/2MB cv0@-256C cpu@48.5C VDD_IN 4629mW/4629mW")
    assert line == TegrastatsLine(rails_mw={"VDD_IN": 4629.0}, temps_c={"cpu": 48.5})


@pytest.mark.parametrize("text", ["", "   ", "NVPM WARN: Golden image context is already created",
                                  "Usage: tegrastats [-option]", "tegrastats: open failed"])
def test_text_that_is_not_a_statistics_line_reads_as_none(text):
    assert parse_tegrastats_line(text) is None


def test_the_power_mode_reads_from_nvpmodel_q_plain_and_verbose():
    assert parse_nvpmodel(recorded_text("jetson_nvpmodel_q.txt")) == "MAXN_SUPER"
    assert parse_nvpmodel(recorded_text("jetson_nvpmodel_q_verbose.txt")) == "15W"


@pytest.mark.parametrize("text", ["", "NVPM ERROR: failed to read PARAM", "bash: nvpmodel: command not found"])
def test_output_that_names_no_power_mode_reads_as_none(text):
    assert parse_nvpmodel(text) is None
