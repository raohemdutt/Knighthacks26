"""The window integral, its two edges, the token-proportional split and the clock-offset arithmetic (domain/energy.py)."""

import itertools
import random

import pytest

from contexts.telemetry.domain.energy import Window, attribute, brackets, clock_offset, integrate


def test_the_window_start_is_clipped_by_interpolation_between_the_samples_around_it():
    # 10 W at t=0 rising to 30 W at t=1: at t=0.5 the line reads 20 W, so [0.5, 1] holds (20 + 30) / 2 * 0.5 J
    trace = [(0.0, 10.0), (1.0, 30.0), (2.0, 30.0)]
    assert integrate(trace, 0.5, 1.0) == pytest.approx(12.5)


def test_the_window_end_is_clipped_by_interpolation_between_the_samples_around_it():
    trace = [(0.0, 10.0), (1.0, 10.0), (2.0, 50.0)]
    # [1, 1.25]: 10 W rising to 20 W at 1.25
    assert integrate(trace, 1.0, 1.25) == pytest.approx((10 + 20) / 2 * 0.25)


def test_both_edges_inside_one_sampling_interval_and_a_window_on_the_samples_themselves():
    trace = [(0.0, 0.0), (4.0, 40.0)]
    assert integrate(trace, 1.0, 3.0) == pytest.approx((10 + 30) / 2 * 2)
    assert integrate(trace, 0.0, 4.0) == pytest.approx(80.0)
    assert integrate(trace, 2.0, 2.0) == 0.0


def test_the_last_sample_at_or_before_the_start_and_the_first_at_or_after_the_end_bracket_the_window():
    times = [0.0, 1.0, 2.0, 3.0]
    assert brackets(times, 1.0, 2.0) == (1, 2)  # at or before / at or after: the samples on the edges themselves
    assert brackets(times, 1.5, 2.5) == (1, 3)
    assert brackets(times, -0.1, 2.0) is None  # nothing at or before the start yet
    assert brackets(times, 1.0, 3.1) is None  # the sample closing the window has not arrived


def test_an_unbracketed_window_has_no_value_rather_than_a_short_one():
    assert integrate([(1.0, 10.0), (2.0, 10.0)], 0.5, 1.5) is None
    assert integrate([(1.0, 10.0), (2.0, 10.0)], 1.5, 2.5) is None


def test_identical_repeated_readings_are_kept_so_a_window_in_a_steady_stretch_is_not_zero():
    # whole watts repeat: 1 Hz samples all reading 27 W. A deduplicating sampler would keep only t=0 and t=10, and an
    # in-window-only integral (Zeus's rule) of [3.2, 5.7] would find no samples at all and read 0 J.
    trace = [(float(t), 27.0) for t in range(11)]
    assert integrate(trace, 3.2, 5.7) == pytest.approx(27.0 * 2.5)
    assert attribute(trace, [Window("r", 3.2, 5.7, 10)]) == {"r": pytest.approx(27.0 * 2.5)}


def test_a_repeated_timestamp_is_a_point_of_the_trace_not_a_stretch():
    trace = [(0.0, 10.0), (1.0, 10.0), (1.0, 30.0), (2.0, 30.0)]
    assert integrate(trace, 0.0, 2.0) == pytest.approx(10.0 + 30.0)


def test_a_request_alone_on_its_node_gets_its_whole_window_integral():
    trace = [(0.0, 20.0), (1.0, 100.0), (2.0, 100.0), (3.0, 20.0)]
    shares = attribute(trace, [Window("r1", 0.4, 2.6, tokens=50)])
    assert shares["r1"] == pytest.approx(integrate(trace, 0.4, 2.6))


def test_overlapping_requests_split_each_stretch_by_output_tokens():
    trace = [(0.0, 100.0), (10.0, 100.0)]  # 100 W flat: 100 J a second
    shares = attribute(trace, [Window("a", 0.0, 4.0, tokens=30), Window("b", 2.0, 6.0, tokens=10)])
    # a alone for [0, 2]: 200 J; both on [2, 4]: 200 J split 3:1; b alone for [4, 6]: 200 J
    assert shares == {"a": pytest.approx(200 + 150), "b": pytest.approx(50 + 200)}


def test_requests_that_report_no_tokens_split_a_shared_stretch_equally():
    trace = [(0.0, 60.0), (10.0, 60.0)]
    shares = attribute(trace, [Window("a", 1.0, 3.0, tokens=None), Window("b", 1.0, 3.0, tokens=0)])
    assert shares == {"a": pytest.approx(60.0), "b": pytest.approx(60.0)}


def test_a_late_request_gets_nothing_from_a_stretch_already_given_out():
    trace = [(0.0, 100.0), (10.0, 100.0)]
    shares = attribute(trace, [Window("old", 0.0, 4.0, tokens=10, fixed=True), Window("late", 2.0, 6.0, tokens=10)])
    assert shares["late"] == pytest.approx(200.0)  # only [4, 6], where nothing was given out yet


def test_a_window_ending_before_it_starts_is_refused():
    with pytest.raises(ValueError):
        integrate([(0.0, 1.0), (2.0, 1.0)], 1.5, 1.0)


@pytest.mark.parametrize("seed", range(5))
def test_the_answers_never_sum_past_the_nodes_energy_over_any_interval(seed):
    """Random overlapping requests on a random trace: for every interval between cut points, the answers whose windows
    lie inside it sum to no more than the node's energy over it, and all answers sum to the node's energy over the
    time at least one ran."""
    rng = random.Random(seed)
    trace = [(t * 0.5, rng.uniform(20, 250)) for t in range(80)]
    windows = []
    for i in range(12):
        start = rng.uniform(0.5, 35)
        windows.append(Window(f"r{i}", start, start + rng.uniform(0.2, 4), tokens=rng.choice([None, 0, 5, 40, 300])))
    shares = attribute(trace, windows)
    assert all(v is not None and v >= 0 for v in shares.values())
    edges = sorted({x for w in windows for x in (w.start, w.end)})
    for lo, hi in itertools.combinations(edges, 2):
        inside = sum(shares[w.key] for w in windows if lo <= w.start and w.end <= hi)
        assert inside <= integrate(trace, lo, hi) + 1e-9
    busy = 0.0  # node energy over the union of the windows
    for a, b in zip(edges, edges[1:]):
        if any(w.start <= a and w.end >= b for w in windows):
            busy += integrate(trace, a, b)
    assert sum(shares.values()) == pytest.approx(busy)


def test_the_offset_is_the_mean_of_the_two_one_way_differences_and_its_error_half_the_round_trip():
    # the pod's clock is 65 s ahead; 0.08 s on the way out, 0.02 s back, 0.001 s on the pod
    t1 = 1000.0
    t2 = t1 + 0.08 + 65.0
    t3 = t2 + 0.001
    t4 = (t3 - 65.0) + 0.02
    offset, round_trip, error = clock_offset(t1, t2, t3, t4)
    assert round_trip == pytest.approx(0.10)
    assert error == pytest.approx(0.05)
    assert offset == pytest.approx(65.0 + (0.08 - 0.02) / 2)
    assert abs(offset - 65.0) <= error


@pytest.mark.parametrize("seed", range(20))
def test_whatever_the_split_of_the_delay_the_true_offset_is_within_the_error(seed):
    rng = random.Random(seed)
    true = rng.uniform(-120, 120)
    out, back, hold = rng.uniform(0, 0.5), rng.uniform(0, 0.5), rng.uniform(0, 0.01)
    t1 = rng.uniform(0, 1e9)
    t2 = t1 + out + true
    t3 = t2 + hold
    t4 = t3 - true + back
    offset, _, error = clock_offset(t1, t2, t3, t4)
    assert abs(offset - true) <= error + 1e-6


def test_an_exchange_that_runs_backwards_is_refused():
    with pytest.raises(ValueError):
        clock_offset(10.0, 75.0, 74.0, 10.2)
