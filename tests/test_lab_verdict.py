"""Behavior of lab.machine.verdict: the numbers a pass or an experiment reads to say whether a change helped.

Where the expected numbers come from (none of them is the code under test):
  * hand arithmetic, written out in the docstring of the test that uses it;
  * printed references: the Student t table, the standard normal table, and the output R documents for
    t.test on its sleep and mtcars data sets;
  * exact closed forms of the t distribution at 1, 2 and 4 degrees of freedom;
  * the exact power of the two-sample t test, from the noncentral t distribution, for the sample-size checks.
"""
import json
import math

import pytest

from lab.machine import verdict as v

# Student t critical values t(0.975, df) and t(0.995, df): the two-sided 5% and 1% columns of a printed table.
T_975 = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
         12: 2.179, 15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042, 40: 2.021, 50: 2.009, 60: 2.000}
T_995 = {2: 9.925, 3: 5.841, 4: 4.604, 5: 4.032, 10: 3.169, 20: 2.845, 30: 2.750, 60: 2.660}

# R's sleep data (extra hours of sleep, ten patients per drug) and mtcars miles per gallon split by transmission.
SLEEP_1 = [0.7, -1.6, -0.2, -1.2, -0.1, 3.4, 3.7, 0.8, 0.0, 2.0]
SLEEP_2 = [1.9, 0.8, 1.1, 0.1, -0.1, 4.4, 5.5, 1.6, 4.6, 3.4]
MPG_AUTOMATIC = [21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 17.8, 16.4, 17.3, 15.2, 10.4, 10.4, 14.7, 21.5,
                 15.5, 15.2, 13.3, 19.2]
MPG_MANUAL = [21.0, 21.0, 22.8, 32.4, 30.4, 33.9, 27.3, 26.0, 30.4, 15.8, 19.7, 15.0, 21.4]

# Five runs per arm, spread 1.5811 in each, B two units above A: Welch collapses to the pooled test, t = 2, df = 8.
RUNS_A = [10, 11, 12, 13, 14]
RUNS_B_NEAR = [12, 13, 14, 15, 16]
RUNS_B_FAR = [15, 16, 17, 18, 19]


# --- the normal distribution ---------------------------------------------------------------------------------

def test_norm_cdf_matches_the_standard_normal_table():
    assert v.norm_cdf(0.0) == 0.5
    assert round(v.norm_cdf(1.0), 4) == 0.8413
    assert round(v.norm_cdf(1.96), 4) == 0.9750
    assert round(v.norm_cdf(2.576), 4) == 0.9950
    assert round(v.norm_cdf(-1.645), 4) == 0.0500


def test_norm_cdf_keeps_the_far_lower_tail():
    """One minus the upper tail would round to zero here; the tail at -8 is 6.2210e-16."""
    assert v.norm_cdf(-8.0) == pytest.approx(6.220960574271785e-16, rel=1e-9)


def test_norm_ppf_gives_the_two_quantiles_the_sample_size_formulas_use():
    assert round(v.norm_ppf(0.975), 6) == 1.959964
    assert round(v.norm_ppf(0.80), 6) == 0.841621
    assert v.norm_ppf(0.5) == 0.0
    assert v.norm_ppf(0.025) == pytest.approx(-v.norm_ppf(0.975), abs=1e-12)


@pytest.mark.parametrize("p", [0.0, 1.0, -0.1, 1.5])
def test_norm_ppf_rejects_probabilities_outside_the_open_interval(p):
    with pytest.raises(ValueError):
        v.norm_ppf(p)


# --- the t distribution --------------------------------------------------------------------------------------

@pytest.mark.parametrize("df,critical", sorted(T_975.items()))
def test_t_cdf_reproduces_the_two_sided_five_percent_table(df, critical):
    """Each printed critical value is rounded to three decimals, so its CDF sits within 1e-4 of 0.975."""
    assert abs(v.t_cdf(critical, df) - 0.975) < 1e-4


@pytest.mark.parametrize("df,critical", sorted(T_995.items()))
def test_t_cdf_reproduces_the_two_sided_one_percent_table(df, critical):
    assert abs(v.t_cdf(critical, df) - 0.995) < 1e-4


@pytest.mark.parametrize("df,critical", sorted(T_975.items()))
def test_t_ppf_reproduces_the_printed_table_at_the_975_quantile(df, critical):
    assert round(v.t_ppf(0.975, df), 3) == critical


@pytest.mark.parametrize("df,critical", sorted(T_995.items()))
def test_t_ppf_reproduces_the_printed_table_at_the_995_quantile(df, critical):
    assert round(v.t_ppf(0.995, df), 3) == critical


def test_t_cdf_matches_the_closed_forms_at_one_two_and_four_degrees_of_freedom():
    """df 1 is the Cauchy law, F(t) = 1/2 + atan(t)/pi. df 2 has F(t) = 1/2 + t / (2 sqrt(2 + t^2)). df 4 has
    F(t) = 1/2 + (3/4) u (1 - u^2 / 3) with u = t / sqrt(t^2 + 4)."""
    for t in (-3.0, -0.5, 0.25, 1.0, 2.5, 10.0):
        assert v.t_cdf(t, 1) == pytest.approx(0.5 + math.atan(t) / math.pi, abs=1e-12)
        assert v.t_cdf(t, 2) == pytest.approx(0.5 + t / (2 * math.sqrt(2 + t * t)), abs=1e-12)
        u = t / math.sqrt(t * t + 4)
        assert v.t_cdf(t, 4) == pytest.approx(0.5 + 0.75 * u * (1 - u * u / 3), abs=1e-12)


def test_t_cdf_is_symmetric_about_zero_and_takes_fractional_degrees_of_freedom():
    assert v.t_cdf(0.0, 7.5) == 0.5
    assert v.t_cdf(-1.7, 7.5) == pytest.approx(1.0 - v.t_cdf(1.7, 7.5), abs=1e-15)
    assert v.t_cdf(2.0, 17.776) > v.t_cdf(2.0, 17.0)  # more degrees of freedom, thinner tails


def test_t_cdf_approaches_the_normal_cdf_for_many_degrees_of_freedom():
    assert abs(v.t_cdf(1.96, 1_000_000) - v.norm_cdf(1.96)) < 1e-5


def test_t_two_sided_p_is_one_at_zero_and_small_in_the_far_tail():
    assert v.t_two_sided_p(0.0, 5) == 1.0
    assert v.t_two_sided_p(-2.228, 10) == pytest.approx(0.05, abs=1e-4)
    assert round(v.t_two_sided_p(5.0, 8), 6) == 0.001053


def test_t_functions_treat_infinite_degrees_of_freedom_as_the_normal_law():
    assert v.t_two_sided_p(1.96, math.inf) == pytest.approx(2 * (1 - v.norm_cdf(1.96)), abs=1e-12)
    assert v.t_cdf(1.96, math.inf) == pytest.approx(v.norm_cdf(1.96), abs=1e-12)


def test_t_two_sided_p_is_zero_for_an_infinite_statistic_and_refuses_nan():
    assert v.t_two_sided_p(math.inf, 5) == 0.0
    assert v.t_two_sided_p(-math.inf, 5) == 0.0
    with pytest.raises(ValueError):
        v.t_two_sided_p(float("nan"), 5)


def test_t_ppf_is_antisymmetric_and_zero_at_the_median():
    assert v.t_ppf(0.5, 12) == 0.0
    assert v.t_ppf(0.025, 10) == pytest.approx(-v.t_ppf(0.975, 10), abs=1e-9)


@pytest.mark.parametrize("df", [0, -3, 0.0])
def test_t_functions_reject_non_positive_degrees_of_freedom(df):
    with pytest.raises(ValueError):
        v.t_cdf(1.0, df)
    with pytest.raises(ValueError):
        v.t_two_sided_p(1.0, df)
    with pytest.raises(ValueError):
        v.t_ppf(0.9, df)


@pytest.mark.parametrize("p", [0.0, 1.0, -0.5, 2.0])
def test_t_ppf_rejects_probabilities_outside_the_open_interval(p):
    with pytest.raises(ValueError):
        v.t_ppf(p, 10)


# --- two proportions -----------------------------------------------------------------------------------------

def test_two_proportion_reproduces_the_hand_checked_reading_of_3_of_60_against_12_of_60():
    """Pooled rate 15/120 = 0.125; se = sqrt(0.125 * 0.875 * (1/60 + 1/60)) = 0.06038; z = (0.20 - 0.05) / 0.06038
    = 2.484; two-sided p = erfc(2.484 / sqrt 2) = 0.0130."""
    z, p = v.two_proportion(3, 60, 12, 60)
    assert round(z, 3) == 2.484
    assert round(p, 4) == 0.013


def test_two_proportion_flips_the_sign_when_the_arms_swap_and_keeps_p():
    z, p = v.two_proportion(3, 60, 12, 60)
    z_swapped, p_swapped = v.two_proportion(12, 60, 3, 60)
    assert z_swapped == pytest.approx(-z, abs=1e-12)
    assert p_swapped == pytest.approx(p, abs=1e-15)


def test_two_proportion_on_a_second_hand_case_50_of_100_against_60_of_100():
    """Pooled 0.55; se = sqrt(0.55 * 0.45 * 0.02) = 0.070356; z = 0.10 / 0.070356 = 1.4213; p = 0.1552."""
    z, p = v.two_proportion(50, 100, 60, 100)
    assert round(z, 3) == 1.421
    assert round(p, 4) == 0.1552


def test_two_proportion_has_no_answer_for_an_arm_with_no_trials():
    assert v.two_proportion(0, 0, 1, 10) == (None, None)
    assert v.two_proportion(1, 10, 0, 0) == (None, None)


def test_two_proportion_sees_nothing_to_tell_apart_when_the_pooled_rate_is_zero_or_one():
    assert v.two_proportion(0, 10, 0, 10) == (0.0, 1.0)
    assert v.two_proportion(10, 10, 10, 10) == (0.0, 1.0)


@pytest.mark.parametrize("args", [(5, 4, 1, 10), (1, 10, 11, 10), (-1, 10, 1, 10), (1, -10, 1, 10)])
def test_two_proportion_rejects_counts_that_cannot_happen(args):
    with pytest.raises(ValueError):
        v.two_proportion(*args)


# --- sample sizes --------------------------------------------------------------------------------------------

def test_trials_per_arm_matches_the_hand_worked_values():
    """5% against 10%, two-sided 5%, power 80%: pooled 0.075; 1.959964 * sqrt(2 * 0.075 * 0.925) = 0.7301;
    0.841621 * sqrt(0.05 * 0.95 + 0.10 * 0.90) = 0.3121; (1.0422)^2 / 0.05^2 = 434.4, so 435 an arm. The same
    arithmetic gives 140.1 (so 141) for 5% against 15%, and 198.96 (so 199) for 10% against 20%."""
    assert v.trials_per_arm(0.05, 0.10) == 435
    assert v.trials_per_arm(0.05, 0.15) == 141
    assert v.trials_per_arm(0.10, 0.20) == 199


def test_trials_per_arm_does_not_care_which_rate_is_listed_first():
    assert v.trials_per_arm(0.10, 0.05) == 435


def test_trials_per_arm_follows_alpha_and_power():
    """Power 90% takes z(0.90) = 1.281552 and gives 581.08 (so 582); alpha 1% takes z(0.995) = 2.575829 and gives
    646.74 (so 647), both for 5% against 10%."""
    assert v.trials_per_arm(0.05, 0.10, power=0.90) == 582
    assert v.trials_per_arm(0.05, 0.10, alpha=0.01) == 647


def test_trials_per_arm_for_a_rate_of_zero_against_one_is_two():
    """Pooled 0.5; 1.959964 * sqrt(2 * 0.25) = 1.3858; the power term is zero; 1.3858^2 / 1 = 1.92."""
    assert v.trials_per_arm(0.0, 1.0) == 2


@pytest.mark.parametrize("p1,p2", [(0.05, 0.05), (-0.1, 0.2), (0.1, 1.1)])
def test_trials_per_arm_needs_two_different_rates_between_zero_and_one(p1, p2):
    with pytest.raises(ValueError):
        v.trials_per_arm(p1, p2)


@pytest.mark.parametrize("kwargs", [{"alpha": 0.0}, {"alpha": 1.0}, {"power": 0.0}, {"power": 1.0}])
def test_trials_per_arm_rejects_alpha_and_power_outside_the_open_interval(kwargs):
    with pytest.raises(ValueError):
        v.trials_per_arm(0.05, 0.10, **kwargs)


@pytest.mark.parametrize("effect,runs", [(3.0, 4), (2.0, 6), (1.5, 9), (1.2, 12), (1.0, 17), (0.8, 26),
                                         (0.5, 64), (0.3, 176)])
def test_runs_per_arm_is_the_smallest_n_with_exact_power_of_eighty_percent(effect, runs):
    """Standardized difference d = delta / sd, two-sided 5%, power 80%. The exact power of the two-sample t test, from
    the noncentral t distribution, at n - 1 and at n: d = 3.0 has 0.7826 at 3 and 0.9389 at 4; d = 2.0 has 0.7905
    at 5 and 0.8764 at 6; d = 1.5 has 0.7965 at 8 and 0.8476 at 9; d = 1.2 has 0.7633 at 11 and 0.8021 at 12; d = 1.0
    has 0.7814 at 16 and 0.8070 at 17; d = 0.8 has 0.7915 at 25 and 0.8075 at 26; d = 0.5 has 0.7952 at 63 and
    0.8015 at 64; d = 0.3 has 0.7991 at 175 and 0.8014 at 176. The widely quoted 64 and 26 for a medium and a large
    effect are in the list. The plain normal formula, 2 (1.96 + 0.84)^2 / d^2, gives 1.7, 3.9, 7.0, 10.9, 15.7, 24.5,
    62.8 and 174.4: two short at the small n a handful of benchmark runs falls in, and one short beyond."""
    assert v.runs_per_arm(1.0, effect) == runs


def test_runs_per_arm_scales_with_the_spread_and_the_difference_together():
    """Only the ratio delta / sd matters: 0.5 / 0.5 is the same d = 1 as 5 / 5."""
    assert v.runs_per_arm(0.5, 0.5) == 17
    assert v.runs_per_arm(5.0, 5.0) == 17


def test_runs_per_arm_relative_turns_a_percentage_of_the_mean_into_the_difference():
    """A mean of 100 and a spread of 5, to see 5%: delta = 5, d = 1, so 17 runs an arm. A mean of 200 and a spread
    of 20, to see 5%: delta = 10, d = 0.5, so 64. The sign of the mean does not matter."""
    assert v.runs_per_arm_relative(sd=5.0, mean=100.0, rel_diff=0.05) == 17
    assert v.runs_per_arm_relative(sd=20.0, mean=200.0, rel_diff=0.05) == 64
    assert v.runs_per_arm_relative(sd=5.0, mean=-100.0, rel_diff=0.05) == 17


def test_runs_per_arm_follows_alpha_and_power():
    """d = 1 at power 90%: 2 (1.96 + 1.28)^2 = 21.0 by the normal approximation, and the t correction gives 23."""
    assert v.runs_per_arm(1.0, 1.0, power=0.90) == 23
    assert v.runs_per_arm(1.0, 1.0, power=0.90) > v.runs_per_arm(1.0, 1.0)
    assert v.runs_per_arm(1.0, 1.0, alpha=0.01) > v.runs_per_arm(1.0, 1.0)


def test_runs_per_arm_with_no_spread_needs_the_two_runs_a_test_requires():
    assert v.runs_per_arm(0.0, 1.0) == 2


def test_runs_per_arm_for_a_tiny_difference_returns_the_normal_approximation_without_hanging():
    """A difference of one millionth of the spread: 2 (2.8016)^2 / 1e-12 = 1.5698e13 runs an arm. The t correction is
    below a hundredth of a percent at that size, so the answer comes from the normal quantiles."""
    assert v.runs_per_arm(1.0, 1e-6) == pytest.approx(1.5698e13, rel=1e-4)


@pytest.mark.parametrize("sd,delta", [(1.0, 0.0), (-1.0, 1.0), (1.0, float("nan")), (float("inf"), 1.0)])
def test_runs_per_arm_rejects_a_missing_difference_and_a_nonsense_spread(sd, delta):
    with pytest.raises(ValueError):
        v.runs_per_arm(sd, delta)


def test_runs_per_arm_relative_rejects_a_zero_relative_difference_and_a_zero_mean():
    with pytest.raises(ValueError):
        v.runs_per_arm_relative(sd=1.0, mean=100.0, rel_diff=0.0)
    with pytest.raises(ValueError):
        v.runs_per_arm_relative(sd=1.0, mean=0.0, rel_diff=0.05)


@pytest.mark.parametrize("kwargs", [{"alpha": 0.0}, {"alpha": 1.0}, {"power": 0.0}, {"power": 1.0}])
def test_runs_per_arm_rejects_alpha_and_power_outside_the_open_interval(kwargs):
    with pytest.raises(ValueError):
        v.runs_per_arm(1.0, 1.0, **kwargs)


# --- the two kinds of arm ------------------------------------------------------------------------------------

def test_samples_from_values_uses_the_sample_standard_deviation():
    """7, 8, 9: mean 8, squared deviations 1 + 0 + 1 = 2, divided by n - 1 = 2 gives variance 1."""
    s = v.Samples.from_values([7.0, 8.0, 9.0])
    assert (s.n, s.mean, s.sd) == (3, 8.0, 1.0)


def test_samples_from_values_has_no_spread_for_fewer_than_two_values():
    assert v.Samples.from_values([4.2]) == v.Samples(n=1, mean=4.2, sd=None)
    assert v.Samples.from_values([]) == v.Samples(n=0, mean=None, sd=None)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_samples_from_values_rejects_values_that_are_not_finite(bad):
    with pytest.raises(ValueError):
        v.Samples.from_values([1.0, bad, 3.0])


@pytest.mark.parametrize("kwargs", [
    {"n": -1, "mean": 1.0, "sd": 1.0},      # negative count
    {"n": 3, "mean": 1.0, "sd": -0.5},      # negative spread
    {"n": 3, "mean": 1.0, "sd": None},      # three runs have a spread
    {"n": 3, "mean": None, "sd": 1.0},      # three runs have a mean
    {"n": 3, "mean": float("nan"), "sd": 1.0},
    {"n": 0, "mean": 1.0, "sd": None},      # no runs, no mean
    {"n": 1, "mean": 1.0, "sd": 0.5},       # one run, no spread
])
def test_samples_rejects_summaries_that_cannot_happen(kwargs):
    with pytest.raises(ValueError):
        v.Samples(**kwargs)


@pytest.mark.parametrize("args", [(5, 4), (-1, 10), (1, -10)])
def test_counts_rejects_counts_that_cannot_happen(args):
    with pytest.raises(ValueError):
        v.Counts(*args)


def test_counts_needs_whole_numbers():
    with pytest.raises(TypeError):
        v.Counts(3.5, 10)


# --- Welch's t test ------------------------------------------------------------------------------------------

def test_welch_reproduces_rs_output_for_the_sleep_data():
    """R documents t.test(extra ~ group, data = sleep): t = -1.8608, df = 17.776, p-value = 0.07939, group means 0.75
    and 2.33. R subtracts group 2 from group 1; this module reads B against A, so with A = group 1 and B = group 2
    the statistic is +1.8608."""
    t, p, df = v.welch(v.Samples.from_values(SLEEP_1), v.Samples.from_values(SLEEP_2))
    assert round(t, 4) == 1.8608
    assert round(df, 3) == 17.776
    assert round(p, 5) == 0.07939


def test_welch_reproduces_rs_output_for_miles_per_gallon_by_transmission():
    """R documents t.test(mpg ~ am, data = mtcars): t = -3.7671, df = 18.332, p-value = 0.001374, group means
    17.14737 (automatic) and 24.39231 (manual). With A = automatic and B = manual the statistic is +3.7671."""
    a = v.Samples.from_values(MPG_AUTOMATIC)
    b = v.Samples.from_values(MPG_MANUAL)
    assert (round(a.mean, 5), round(b.mean, 5)) == (17.14737, 24.39231)
    t, p, df = v.welch(a, b)
    assert round(t, 4) == 3.7671
    assert round(df, 3) == 18.332
    assert round(p, 6) == 0.001374


def test_welch_with_equal_sizes_and_spreads_is_the_pooled_test_on_four_degrees_of_freedom():
    """A = 1, 2, 3 and B = 3, 4, 5: both spreads are 1, the difference is 2, se = sqrt(1/3 + 1/3) = 0.8165, so
    t = sqrt(6) = 2.4495 and Welch's df collapses to 3 + 3 - 2 = 4. At 4 df, F(t) = 1/2 + (3/4) u (1 - u^2/3) with
    u = t / sqrt(t^2 + 4) = sqrt(0.6), so F = 0.964758 and the two-sided p = 0.070484."""
    t, p, df = v.welch(v.Samples.from_values([1, 2, 3]), v.Samples.from_values([3, 4, 5]))
    assert t == pytest.approx(math.sqrt(6), abs=1e-12)
    assert df == pytest.approx(4.0, abs=1e-12)
    assert round(p, 6) == 0.070484


def test_welch_degrees_of_freedom_by_hand_for_unequal_spreads():
    """A = 1, 2, 3 (mean 2, variance 1) and B = 2, 4, 6 (mean 4, variance 4): the variances of the means are 1/3 and
    4/3, se^2 = 5/3, t = 2 / sqrt(5/3) = 1.549193 and df = (5/3)^2 / ((1/3)^2 / 2 + (4/3)^2 / 2) = 50/17."""
    t, p, df = v.welch(v.Samples.from_values([1, 2, 3]), v.Samples.from_values([2, 4, 6]))
    assert t == pytest.approx(2 / math.sqrt(5 / 3), abs=1e-12)
    assert df == pytest.approx(50 / 17, abs=1e-12)
    assert round(p, 5) == 0.22088


def test_welch_with_one_steady_arm_takes_the_degrees_of_freedom_of_the_noisy_one():
    """A repeats 5 four times, so only B contributes variance and the df is B's n - 1 = 3. B = 6, 7, 8, 7 has mean
    7 and variance 2/3, so se^2 = (2/3) / 4 and t = 2 / sqrt(1/6) = 2 sqrt(6) = 4.898979. At 3 df,
    F(t) = 1/2 + (atan(s) + s / (1 + s^2)) / pi with s = t / sqrt(3), so p = 0.016277."""
    t, p, df = v.welch(v.Samples.from_values([5, 5, 5, 5]), v.Samples.from_values([6, 7, 8, 7]))
    assert t == pytest.approx(2 * math.sqrt(6), abs=1e-12)
    assert df == pytest.approx(3.0, abs=1e-12)
    assert round(p, 6) == 0.016277


def test_welch_flips_the_sign_when_the_arms_swap_and_keeps_p_and_df():
    a, b = v.Samples.from_values(SLEEP_1), v.Samples.from_values(SLEEP_2)
    t, p, df = v.welch(a, b)
    t_swapped, p_swapped, df_swapped = v.welch(b, a)
    assert t_swapped == pytest.approx(-t, abs=1e-12)
    assert p_swapped == pytest.approx(p, abs=1e-12)
    assert df_swapped == pytest.approx(df, abs=1e-12)


def test_welch_accepts_summaries_as_well_as_raw_runs():
    raw = v.welch(v.Samples.from_values(RUNS_A), v.Samples.from_values(RUNS_B_NEAR))
    summary = v.welch(v.Samples(n=5, mean=12.0, sd=math.sqrt(2.5)), v.Samples(n=5, mean=14.0, sd=math.sqrt(2.5)))
    assert summary == pytest.approx(raw, abs=1e-12)


@pytest.mark.parametrize("a,b", [([5.0], [1.0, 2.0, 3.0]), ([1.0, 2.0, 3.0], [5.0]), ([], [1.0, 2.0])])
def test_welch_has_no_answer_when_an_arm_has_fewer_than_two_runs(a, b):
    assert v.welch(v.Samples.from_values(a), v.Samples.from_values(b)) == (None, None, None)


def test_welch_calls_two_arms_that_repeat_the_same_value_equal_and_tests_nothing_else():
    same = v.welch(v.Samples.from_values([5.0] * 5), v.Samples.from_values([5.0] * 5))
    assert same == (0.0, 1.0, None)


def test_welch_cannot_test_two_steady_arms_that_differ():
    """With no spread in either arm the run-to-run noise is unknown, so a difference cannot be called real."""
    assert v.welch(v.Samples.from_values([7.7] * 5), v.Samples.from_values([8.1] * 5)) == (None, None, None)


# --- the verdict ---------------------------------------------------------------------------------------------

def test_the_four_labels_are_the_documented_strings():
    assert v.LABELS == ("too_early", "no_difference", "b_ahead", "a_ahead")
    assert (v.TOO_EARLY, v.NO_DIFFERENCE, v.B_AHEAD, v.A_AHEAD) == v.LABELS


def test_verdict_calls_3_of_60_against_12_of_60_b_ahead_with_the_hand_checked_statistic():
    r = v.verdict(v.Counts(3, 60), v.Counts(12, 60))
    assert r.label == "b_ahead"
    assert r.test == "two_proportion"
    assert round(r.statistic, 3) == 2.484
    assert round(r.p, 4) == 0.013
    assert r.df is None
    assert r.diff == pytest.approx(0.15, abs=1e-12)
    assert (r.n_a, r.n_b, r.min_per_arm) == (60, 60, 50)
    assert (r.needed_per_arm, r.more_a, r.more_b) == (None, 0, 0)
    assert r.note == "B ahead of A: 20.0% against 5.0% (z = 2.484, p = 0.013)"


def test_verdict_names_a_when_a_is_the_higher_rate():
    r = v.verdict(v.Counts(12, 60), v.Counts(3, 60))
    assert r.label == "a_ahead"
    assert round(r.statistic, 3) == -2.484
    assert r.note == "A ahead of B: 20.0% against 5.0% (z = -2.484, p = 0.013)"


def test_verdict_names_the_lower_rate_when_lower_is_better():
    r = v.verdict(v.Counts(12, 60), v.Counts(3, 60), higher_is_better=False)
    assert r.label == "b_ahead"
    assert round(r.statistic, 3) == -2.484
    assert r.note == "B ahead of A: 5.0% against 20.0% (z = -2.484, p = 0.013; lower is better)"


def test_verdict_calls_3_of_60_against_4_of_60_no_difference_and_says_how_many_more_would_tell():
    """Pooled 7/120, so z squared = (1/60)^2 / (7/120 * 113/120 * 1/30) = 120/791 exactly and z = 0.3895, p = 0.697.
    Telling 5.0% from 6.7% needs 3103.05, so 3104 trials an arm, which is 3044 more than the 60 each has."""
    r = v.verdict(v.Counts(3, 60), v.Counts(4, 60))
    assert r.label == "no_difference"
    assert r.statistic ** 2 == pytest.approx(120 / 791, abs=1e-12)
    assert round(r.p, 3) == 0.697
    assert r.needed_per_arm == 3104
    assert (r.more_a, r.more_b) == (3044, 3044)
    assert r.note == ("no difference shown: B 6.7% against A 5.0% (z = 0.389, p = 0.6969); a difference this size "
                      "needs 3104 trials per arm at 80% power, 3044 more each")


def test_verdict_with_equal_observed_rates_has_no_sample_size_to_offer():
    r = v.verdict(v.Counts(30, 60), v.Counts(30, 60))
    assert r.label == "no_difference"
    assert (r.statistic, r.p, r.diff) == (0.0, 1.0, 0.0)
    assert (r.needed_per_arm, r.more_a, r.more_b) == (None, 0, 0)
    assert r.note.endswith("the observed values are equal, so no sample size follows from them")


def test_verdict_is_too_early_under_fifty_trials_whatever_the_counts_say():
    """0 of 10 against 10 of 10 has z = 4.47, but ten trials an arm is under the floor of 50: 40 more each."""
    r = v.verdict(v.Counts(0, 10), v.Counts(10, 10))
    assert r.label == "too_early"
    assert round(r.statistic, 3) == 4.472
    assert r.p < 1e-4
    assert (r.min_per_arm, r.needed_per_arm, r.more_a, r.more_b) == (50, 2, 40, 40)
    assert r.note == "too early to call: A has 10 trials (0 successes), B has 10 (10); 50 each needed, 40 more each"


def test_verdict_too_early_reports_the_larger_of_the_floor_and_the_size_the_observed_gap_needs():
    """1 of 10 against 2 of 10 is 10% against 20%, which needs 199 trials an arm: 189 more each, not just 40."""
    r = v.verdict(v.Counts(1, 10), v.Counts(2, 10))
    assert r.label == "too_early"
    assert (r.needed_per_arm, r.more_a, r.more_b) == (199, 189, 189)
    assert "199 each needed, 189 more each" in r.note


def test_verdict_too_early_counts_each_arm_separately_when_the_arms_differ_in_size():
    """60 trials is above the floor of 50 and 20 is under it: A needs nothing more, B needs 30. 3 of 60 against 12
    of 20 needs only 11 trials an arm to be told apart, so the floor decides."""
    r = v.verdict(v.Counts(3, 60), v.Counts(12, 20))
    assert r.label == "too_early"
    assert (r.needed_per_arm, r.more_a, r.more_b) == (11, 0, 30)
    assert "0 more for A, 30 more for B" in r.note


def test_verdict_with_an_arm_that_has_no_trials_is_too_early_and_has_no_rate_to_size_on():
    r = v.verdict(v.Counts(0, 0), v.Counts(5, 50))
    assert r.label == "too_early"
    assert (r.statistic, r.p, r.diff, r.needed_per_arm) == (None, None, None, None)
    assert (r.more_a, r.more_b) == (50, 0)
    assert "A has 0 trials (0 successes), B has 50 (5)" in r.note


def test_verdict_floor_is_the_callers_to_lower():
    """The same ten against ten, with the floor lowered to 10, is read: p = 7.7e-6, so B ahead."""
    r = v.verdict(v.Counts(0, 10), v.Counts(10, 10), min_per_arm=10)
    assert r.label == "b_ahead"
    assert r.min_per_arm == 10
    assert (r.more_a, r.more_b) == (0, 0)


def test_verdict_alpha_moves_the_line_between_ahead_and_no_difference():
    """3 of 60 against 12 of 60 has p = 0.01298: ahead at alpha 0.05, not at alpha 0.01."""
    assert v.verdict(v.Counts(3, 60), v.Counts(12, 60), alpha=0.05).label == "b_ahead"
    assert v.verdict(v.Counts(3, 60), v.Counts(12, 60), alpha=0.01).label == "no_difference"


def test_verdict_reads_the_sleep_study_as_no_difference_at_five_percent():
    """R's p-value for the sleep data is 0.07939, above 0.05, so the longer sleep with drug 2 is not called real."""
    r = v.verdict(SLEEP_1, SLEEP_2)
    assert r.label == "no_difference"
    assert r.test == "welch"
    assert round(r.statistic, 4) == 1.8608
    assert round(r.df, 3) == 17.776
    assert round(r.p, 5) == 0.07939
    assert r.diff == pytest.approx(1.58, abs=1e-12)
    assert (r.n_a, r.n_b, r.min_per_arm) == (10, 10, 5)
    assert r.needed_per_arm > 10 and r.more_a == r.more_b == r.needed_per_arm - 10


def test_verdict_reads_miles_per_gallon_by_transmission_as_b_ahead_when_higher_is_better():
    r = v.verdict(MPG_AUTOMATIC, MPG_MANUAL)
    assert r.label == "b_ahead"
    assert round(r.statistic, 4) == 3.7671
    assert round(r.p, 6) == 0.001374
    assert (r.n_a, r.n_b) == (19, 13)
    assert (r.needed_per_arm, r.more_a, r.more_b) == (None, 0, 0)
    assert r.note == "B ahead of A: mean 24.39 against mean 17.15 (t = 3.767, df = 18.3, p = 0.0014)"


def test_verdict_names_the_lower_mean_when_lower_is_better():
    """Joules per token: B burns less, so with lower being better B is ahead; with higher being better, A is."""
    a = [26.0, 26.5, 25.8, 26.2, 26.1, 26.4]
    b = [24.9, 25.1, 24.8, 25.2, 25.0, 25.3]
    assert v.verdict(a, b, higher_is_better=False).label == "b_ahead"
    assert v.verdict(a, b, higher_is_better=True).label == "a_ahead"
    r = v.verdict(a, b, higher_is_better=False)
    assert round(r.statistic, 3) == -8.578
    assert round(r.df, 2) == 9.12
    assert r.p < 1e-4
    assert r.note == "B ahead of A: mean 25.05 against mean 26.17 (t = -8.578, df = 9.1, p < 0.0001; lower is better)"


def test_verdict_on_runs_with_t_equal_to_two_is_no_difference_and_sizes_the_next_batch():
    """Runs 10..14 against 12..16: both spreads 1.5811, se = 1, t = 2.0 on 8 df, p = 0.0805. The gap of 2 is
    d = 1.2649 spreads, which needs 11 runs an arm for power 80% (power 0.763 at 10, 0.806 at 11): 6 more each."""
    r = v.verdict(RUNS_A, RUNS_B_NEAR)
    assert r.label == "no_difference"
    assert (round(r.statistic, 6), round(r.df, 6), round(r.p, 4)) == (2.0, 8.0, 0.0805)
    assert (r.needed_per_arm, r.more_a, r.more_b) == (11, 6, 6)
    assert r.note == ("no difference shown: B mean 14 against A mean 12 (t = 2.000, df = 8.0, p = 0.0805); a "
                      "difference this size needs 11 runs per arm at 80% power, 6 more each")


def test_verdict_on_runs_five_units_apart_is_b_ahead():
    """Runs 10..14 against 15..19: se = 1, t = 5.0 on 8 df, p = 0.001053."""
    r = v.verdict(RUNS_A, RUNS_B_FAR)
    assert r.label == "b_ahead"
    assert (round(r.statistic, 6), round(r.df, 6), round(r.p, 6)) == (5.0, 8.0, 0.001053)
    assert r.note == "B ahead of A: mean 17 against mean 12 (t = 5.000, df = 8.0, p = 0.0011)"


def test_verdict_takes_summaries_and_gives_the_same_answer_as_the_raw_runs():
    raw = v.verdict(RUNS_A, RUNS_B_NEAR)
    summary = v.verdict(v.Samples(n=5, mean=12.0, sd=math.sqrt(2.5)), v.Samples(n=5, mean=14.0, sd=math.sqrt(2.5)))
    assert summary.label == raw.label and summary.note == raw.note
    assert summary.statistic == pytest.approx(raw.statistic, abs=1e-12)
    assert (summary.needed_per_arm, summary.more_a, summary.more_b) == (raw.needed_per_arm, raw.more_a, raw.more_b)


def test_verdict_on_runs_is_too_early_under_five_runs_an_arm():
    """Four runs an arm: the floor for runs is 5, so one more each, though t = 5.89 on 5.07 df has p = 0.0019."""
    r = v.verdict([7.6, 7.8, 7.7, 7.7], [8.1, 8.3, 8.0, 8.2])
    assert r.label == "too_early"
    assert round(r.statistic, 3) == 5.892
    assert round(r.p, 4) == 0.0019
    assert (r.min_per_arm, r.needed_per_arm, r.more_a, r.more_b) == (5, 3, 1, 1)
    assert r.note == "too early to call: A has 4 runs (mean 7.7), B has 4 (mean 8.15); 5 each needed, 1 more each"
    assert v.verdict([7.6, 7.8, 7.7, 7.7], [8.1, 8.3, 8.0, 8.2], min_per_arm=4).label == "b_ahead"


def test_verdict_on_a_single_run_is_too_early_and_asks_for_the_floor():
    r = v.verdict([7.7], [8.1, 8.2, 8.0, 8.1, 8.3])
    assert r.label == "too_early"
    assert (r.statistic, r.p, r.df) == (None, None, None)
    assert (r.more_a, r.more_b) == (4, 0)
    assert r.note == ("too early to call: A has 1 run (mean 7.7), B has 5 (mean 8.14); 5 each needed, "
                      "4 more for A, 0 more for B")


def test_verdict_note_says_one_trial_and_one_success_in_the_singular():
    r = v.verdict(v.Counts(1, 1), v.Counts(0, 1))
    assert r.note == "too early to call: A has 1 trial (1 success), B has 1 (0); 50 each needed, 49 more each"


def test_verdict_sizes_the_next_batch_at_the_power_it_is_given():
    """The power moves the size, and the note names it; the sizes themselves are checked against hand values in the
    sample-size tests above."""
    rates = v.verdict(v.Counts(3, 60), v.Counts(4, 60), power=0.90)
    assert rates.needed_per_arm == v.trials_per_arm(0.05, 4 / 60, power=0.90) > 3104
    assert "at 90% power" in rates.note
    runs = v.verdict(RUNS_A, RUNS_B_NEAR, power=0.90)
    assert runs.needed_per_arm == v.runs_per_arm(math.sqrt(2.5), 2.0, power=0.90) > 11
    assert "at 90% power" in runs.note


def test_verdict_on_two_steady_arms_that_differ_is_too_early_and_cannot_say_how_many_more():
    """Five identical runs in each arm give no estimate of the noise, and more of the same would not either."""
    r = v.verdict([7.7] * 5, [8.1] * 5)
    assert r.label == "too_early"
    assert (r.statistic, r.p, r.df, r.needed_per_arm, r.more_a, r.more_b) == (None, None, None, None, None, None)
    assert r.note == ("too early to call: both arms repeat one value (7.7 and 8.1), so the run-to-run noise cannot "
                      "be estimated")


def test_verdict_on_two_steady_arms_that_agree_is_no_difference():
    r = v.verdict([5.0] * 5, [5.0] * 5)
    assert r.label == "no_difference"
    assert (r.statistic, r.p, r.df, r.diff) == (0.0, 1.0, None, 0.0)
    assert (r.needed_per_arm, r.more_a, r.more_b) == (None, 0, 0)


def test_verdict_floor_for_runs_cannot_be_below_two():
    with pytest.raises(ValueError):
        v.verdict(RUNS_A, RUNS_B_NEAR, min_per_arm=1)


@pytest.mark.parametrize("floor", [0, -5, 2.5])
def test_verdict_floor_for_counts_must_be_a_whole_number_of_at_least_one(floor):
    with pytest.raises(ValueError):
        v.verdict(v.Counts(3, 60), v.Counts(12, 60), min_per_arm=floor)


@pytest.mark.parametrize("kwargs", [{"alpha": 0.0}, {"alpha": 1.0}, {"power": 0.0}, {"power": 1.0}])
def test_verdict_rejects_alpha_and_power_outside_the_open_interval(kwargs):
    with pytest.raises(ValueError):
        v.verdict(v.Counts(3, 60), v.Counts(12, 60), **kwargs)
    with pytest.raises(ValueError):
        v.verdict(RUNS_A, RUNS_B_NEAR, **kwargs)


def test_verdict_refuses_to_compare_a_rate_with_a_set_of_runs():
    with pytest.raises(TypeError):
        v.verdict(v.Counts(3, 60), RUNS_A)
    with pytest.raises(TypeError):
        v.verdict(RUNS_A, v.Counts(3, 60))


@pytest.mark.parametrize("bad", [42, None, "abc"])
def test_verdict_refuses_an_arm_that_is_neither_counts_nor_runs(bad):
    with pytest.raises(TypeError):
        v.verdict(bad, RUNS_A)


def test_as_dict_is_plain_json_with_every_field():
    r = v.verdict(v.Counts(3, 60), v.Counts(12, 60))
    d = r.as_dict()
    assert list(d) == ["label", "test", "statistic", "p", "df", "diff", "n_a", "n_b", "min_per_arm",
                       "needed_per_arm", "more_a", "more_b", "note"]
    assert json.loads(json.dumps(d)) == d
    assert d["label"] == "b_ahead" and d["note"] == r.note


def test_as_dict_keeps_the_gaps_as_nulls():
    d = v.verdict([7.7] * 5, [8.1] * 5).as_dict()
    assert json.loads(json.dumps(d)) == d
    assert (d["statistic"], d["p"], d["df"], d["more_a"], d["more_b"]) == (None, None, None, None, None)
