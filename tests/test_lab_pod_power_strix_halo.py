"""lab/pod_power.py on the SJC pods' Strix Halo (Radeon 8060S, gfx1151), from tool output recorded on pod C.

The strixhalo_*.txt fixtures are what /opt/rocm/bin/amd-smi and rocm-smi printed on the account C pod on
2026-10-10 (docs/podc-smi-sample-2026-10-10.txt in the workspace), one file per command. On that image amd-smi is
not on PATH, `amd-smi metric --power` reads SOCKET_POWER, `--energy` and `--clock` read N/A, `--temperature` reads
only EDGE, and rocm-smi labels its power "Current Socket Graphics Package Power". The R9700 fixtures beside them are
the other chip's, kept for the parsers.
"""

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from contracts import parse_event
from contracts.telemetry import GpuPower
from lab import pod_power
from lab.pod_power import PodSampler

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "telemetry" / "pod_power"
T0 = datetime(2026, 10, 10, 19, 48, tzinfo=UTC).timestamp()


def fixture(name: str) -> str:
    return (FIXTURES / f"strixhalo_{name}.txt").read_text(encoding="utf-8")


AMD_POWER = fixture("amdsmi_metric_power")
AMD_ENERGY = fixture("amdsmi_metric_energy")
AMD_CLOCK_TEMP = fixture("amdsmi_metric_clock_temperature")
ROCM_POWER = fixture("rocmsmi_showpower")
ROCM_CLOCKS = fixture("rocmsmi_showclocks")
ROCM_TEMP = fixture("rocmsmi_showtemp")


class Tools:
    """What each command line printed on the pod; any other command line fails as an unknown option would."""

    def __init__(self, answers: dict[str, str]) -> None:
        self.answers, self.asked = answers, []

    def run(self, argv):
        command = " ".join([Path(argv[0]).name, *argv[1:]])
        self.asked.append(command)
        if command not in self.answers:
            raise pod_power.ReadError(f"{argv[0]} exited 2: no recorded answer for {command}")
        return self.answers[command]

    @staticmethod
    def which(name):
        return f"/opt/rocm/bin/{name}"  # where find_tool finds them on this image, which has neither on PATH


# --- the parsers on the recorded output ----------------------------------------------------------------------------

def test_amd_smi_power_on_the_strix_halo_reads_socket_power_and_nothing_it_did_not_print():
    reading = pod_power.parse_amdsmi(AMD_POWER)[0]
    assert (reading.socket_w, reading.power_field) == (110.0, "socket_power")
    assert (reading.gfx_mhz, reading.mem_mhz, reading.throttled, reading.energy_j) == (None, None, None, None)
    assert reading.temps_c == {}


def test_amd_smi_energy_and_clock_read_n_a_on_the_strix_halo_and_its_temperature_is_the_edge_only():
    assert pod_power.parse_amdsmi(AMD_ENERGY)[0].energy_j is None
    reading = pod_power.parse_amdsmi(AMD_CLOCK_TEMP)[0]
    assert (reading.gfx_mhz, reading.mem_mhz) == (None, None)
    assert reading.temps_c == {"edge": 66.0}


def test_rocm_smi_on_the_strix_halo_reads_socket_power_the_clocks_and_the_edge_sensor():
    power = pod_power.parse_rocmsmi(ROCM_POWER)[0]
    assert (power.socket_w, power.power_field) == (110.038, "current_socket_graphics_package_power")
    clocks = pod_power.parse_rocmsmi(ROCM_CLOCKS)[0]
    assert (clocks.gfx_mhz, clocks.mem_mhz) == (2900.0, 1000.0)
    assert pod_power.parse_rocmsmi(ROCM_TEMP)[0].temps_c == {"edge": 66.0}


# --- choosing the reader -------------------------------------------------------------------------------------------

def test_amd_smi_power_alone_is_completed_with_rocm_smis_clocks_and_temperature():
    tools = Tools({"amd-smi metric --power": AMD_POWER,
                   "rocm-smi --showtemp --showclocks": ROCM_TEMP + ROCM_CLOCKS})

    reader, attempts = pod_power.choose_reader(run=tools.run, which=tools.which)

    assert (reader.source, tuple(reader.options)) == ("amd-smi", ("metric", "--power"))
    reading = reader.read()
    assert (reading.socket_w, reading.power_field) == (110.0, "socket_power")  # power stays amd-smi's
    assert (reading.gfx_mhz, reading.mem_mhz, reading.temps_c) == (2900.0, 1000.0, {"edge": 66.0})
    assert reader.describe()["fill"] == {"tool": "rocm-smi", "options": ["--showtemp", "--showclocks"], "gpu": 0,
                                         "fields": ["gfx_mhz", "mem_mhz", "temps_c"]}
    assert attempts[-1]["tool"] == "rocm-smi" and attempts[-1]["fills"] == ["gfx_mhz", "mem_mhz", "temps_c"]


def test_an_amd_smi_read_that_has_its_own_clocks_and_temperatures_is_not_filled():
    r9700 = (FIXTURES / "amdsmi_metric_idle.txt").read_text(encoding="utf-8")
    tools = Tools({"amd-smi metric --power --clock --temperature --usage --energy": r9700})

    reader, _ = pod_power.choose_reader(run=tools.run, which=tools.which)

    assert "fill" not in reader.describe()
    assert tools.asked == ["amd-smi metric --power --clock --temperature --usage --energy"]


def test_a_fill_read_that_fails_later_keeps_the_power_sample():
    def run(argv):
        if argv[0].endswith("rocm-smi"):
            raise pod_power.ReadError("rocm-smi exited 1: busy")
        return AMD_POWER

    main = pod_power.CommandReader("amd-smi", "/opt/rocm/bin/amd-smi", ("metric", "--power"), pod_power.parse_amdsmi,
                                   run=run)
    fill = pod_power.CommandReader("rocm-smi", "/opt/rocm/bin/rocm-smi", pod_power.ROCMSMI_FILL_OPTIONS,
                                   pod_power.parse_rocmsmi, run=run)
    reading = pod_power.FilledReader(main, fill).read()
    assert (reading.socket_w, reading.gfx_mhz, reading.temps_c) == (110.0, None, {})


# --- the sampler and the energy counter on this card ---------------------------------------------------------------

def strix_reading(watts: float):
    reading = pod_power.parse_amdsmi(AMD_POWER)[0]
    filled = pod_power.parse_rocmsmi(ROCM_TEMP + ROCM_CLOCKS)[0]
    return pod_power.Reading(socket_w=watts, gfx_mhz=filled.gfx_mhz, mem_mhz=filled.mem_mhz,
                             temps_c=dict(filled.temps_c), energy_j=pod_power.parse_amdsmi(AMD_ENERGY)[0].energy_j,
                             power_field=reading.power_field)


def test_without_an_energy_counter_the_checks_say_so_and_energy_always_comes_from_integrated_power():
    s = PodSampler(SimpleNamespace(source="amd-smi"), "gpu-a", clock=lambda: T0)
    events = [s.record(strix_reading(27.0 + (n % 3)), T0 + n) for n in range(12)]  # idle: the start-up check
    events += [s.record(strix_reading(110.0 + (n % 5)), T0 + 12 + n) for n in range(30)]  # load: the load check

    assert [c["phase"] for c in s.checks] == ["start-up", "load"]
    assert all(c["advanced"] is None and c["agrees"] is False for c in s.checks)
    assert s.counter_ok is False
    assert all(isinstance(parse_event(e), GpuPower) for e in events)
    summary = pod_power.summarize(events)
    assert (summary["energy_source"], summary["counter_ok"]) == ("avg_power_integral", False)
    assert summary["start_temps_c"] == {"edge": 66.0}


def test_load_by_power_fits_the_strix_halo_idle_and_load():
    s = PodSampler(SimpleNamespace(source="amd-smi"), "gpu-a", clock=lambda: T0)
    assert s.record(strix_reading(29.0), T0)["under_load"] is False  # the 24 to 29 W idle of these pods
    assert s.record(strix_reading(80.0), T0 + 1)["under_load"] is True  # well under its 110 W, still busy


# --- the node gate -------------------------------------------------------------------------------------------------

STRIX_HALO_ROCMINFO = "  Marketing Name:          AMD Radeon 8060S Graphics\n  Name:                    gfx1151\n"
R9700_ROCMINFO = "  Marketing Name:          AMD Radeon AI PRO R9700\n  Name:                    gfx1201\n"
RX_9070_XT_ROCMINFO = "  Marketing Name:          AMD Radeon RX 9070 XT\n  Name:                    gfx1201\n"


def test_the_gate_admits_the_strix_halo_and_refuses_the_other_chips():
    assert pod_power.gate("gpu-a", lambda: STRIX_HALO_ROCMINFO) is None
    for other in (R9700_ROCMINFO, RX_9070_XT_ROCMINFO, "  Name:   gfx11510\n"):
        assert pod_power.gate("gpu-b", lambda: other) == "rocminfo names no gfx1151: this is not a Strix Halo pod"
