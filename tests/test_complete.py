"""Tests for the completion pass: the road network, the learned ETA, the surge
controller, and repositioning."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import control as CTL      # noqa: E402
from src import eta_model as EM     # noqa: E402
from src import roadnet as RN       # noqa: E402


# --------------------------------------------------------------------------
# the road network
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def net():
    return RN.RoadNetwork(n_rows=25, n_cols=25, seed=1)


def test_a_route_is_longer_than_the_straight_line(net):
    """The correction every haversine ETA was missing."""
    a = net.node_latlon(2, 3)
    b = net.node_latlon(20, 18)
    r = net.route(a[0], a[1], b[0], b[1])
    assert r["metres"] >= r["straight_metres"]
    assert r["detour_factor"] >= 1.0


def test_routing_does_not_silently_fall_back_to_haversine(net):
    """A routing bug that returns haversine looks exactly like a working router
    on a featureless grid. This is the guard that caught it: the first version
    had one-ways restricting travel ACROSS a street rather than along it, which
    stranded most of the grid and made every route a fallback."""
    rng = np.random.default_rng(0)
    fallbacks = 0
    for _ in range(60):
        a = net.node_latlon(int(rng.integers(0, net.R)), int(rng.integers(0, net.C)))
        b = net.node_latlon(int(rng.integers(0, net.R)), int(rng.integers(0, net.C)))
        fallbacks += int(net.route(a[0], a[1], b[0], b[1]).get("unreachable", False))
    assert fallbacks == 0


def test_rush_hour_costs_time_and_haversine_has_no_clock(net):
    a = net.node_latlon(1, 1)
    b = net.node_latlon(22, 20)
    quiet = net.route(a[0], a[1], b[0], b[1], hour=3.0)["seconds"]
    peak = net.route(a[0], a[1], b[0], b[1], hour=8.5)["seconds"]
    assert peak > quiet * 1.3


def test_one_ways_break_symmetry(net):
    """haversine guarantees d(a,b) == d(b,a). A scorer that caches 'the distance
    between a and b' is wrong half the time once that stops holding."""
    rng = np.random.default_rng(3)
    asym = 0
    for _ in range(40):
        a = net.node_latlon(int(rng.integers(0, net.R)), int(rng.integers(0, net.C)))
        b = net.node_latlon(int(rng.integers(0, net.R)), int(rng.integers(0, net.C)))
        f = net.route(a[0], a[1], b[0], b[1])["seconds"]
        r = net.route(b[0], b[1], a[0], a[1])["seconds"]
        if abs(f - r) > 1.0:
            asym += 1
    assert asym > 0


def test_a_route_to_itself_is_free(net):
    a = net.node_latlon(5, 5)
    r = net.route(a[0], a[1], a[0], a[1])
    assert r["seconds"] == 0.0


def test_the_detour_profile_reports_a_distribution_not_just_a_mean(net):
    """A mean detour factor can be applied as a multiplier; the tail cannot, and
    the tail is where the ETAs blow up."""
    prof = RN.detour_profile(net, n=80, seed=4)
    assert prof["p99"] >= prof["median"] >= 1.0
    assert prof["mean"] > 1.05


# --------------------------------------------------------------------------
# the ETA models
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def eta_data(net):
    X, y, meta = EM.generate(net, n=1200, n_restaurants=20, seed=2)
    return X, y, meta


def test_the_analytic_baseline_is_not_a_straw_man(eta_data):
    """It is given every ADDITIVE term the generator uses, so the comparison
    isolates the interaction rather than a term anybody could configure."""
    X, y, _ = eta_data
    pred = EM.analytic_eta(X[:, 0], X[:, 7], X[:, 8], items=X[:, 5])
    ev = EM.evaluate(pred, y)
    assert abs(ev["bias"]) < 3.0, ev


def test_the_learned_model_beats_it_on_mae(eta_data):
    X, y, _ = eta_data
    cut = int(0.7 * len(X))
    m = EM.fit_learned(X[:cut], y[:cut], n_estimators=120)
    learned = EM.evaluate(m.predict(X[cut:]), y[cut:])
    analytic = EM.evaluate(
        EM.analytic_eta(X[cut:, 0], X[cut:, 7], X[cut:, 8], items=X[cut:, 5]),
        y[cut:])
    assert learned["mae"] < analytic["mae"]


def test_a_mean_eta_is_late_about_half_the_time(eta_data):
    """By construction, which is the whole argument for shipping a quantile."""
    X, y, _ = eta_data
    cut = int(0.7 * len(X))
    m = EM.fit_learned(X[:cut], y[:cut], n_estimators=120)
    ev = EM.evaluate(m.predict(X[cut:]), y[cut:])
    assert 0.40 < ev["late_share"] < 0.60


def test_the_p80_is_late_about_a_fifth_of_the_time(eta_data):
    X, y, _ = eta_data
    cut = int(0.7 * len(X))
    m = EM.fit_learned(X[:cut], y[:cut], quantile=0.80, n_estimators=120)
    ev = EM.evaluate(m.predict(X[cut:]), y[cut:])
    assert 0.10 < ev["late_share"] < 0.35


def test_the_p80_is_worse_on_mae_and_that_is_the_point(eta_data):
    """Choosing an ETA model on MAE optimises a quantity nobody experiences."""
    X, y, _ = eta_data
    cut = int(0.7 * len(X))
    mean_m = EM.fit_learned(X[:cut], y[:cut], n_estimators=120)
    q_m = EM.fit_learned(X[:cut], y[:cut], quantile=0.80, n_estimators=120)
    mean_ev = EM.evaluate(mean_m.predict(X[cut:]), y[cut:])
    q_ev = EM.evaluate(q_m.predict(X[cut:]), y[cut:])
    assert q_ev["mae"] > mean_ev["mae"]
    assert q_ev["late_share"] < mean_ev["late_share"]


# --------------------------------------------------------------------------
# the surge controller
# --------------------------------------------------------------------------
def test_hysteresis_requires_a_gap():
    """A controller that claims hysteresis and has a single threshold is worse
    than one that never claimed it."""
    with pytest.raises(ValueError):
        CTL.SurgeController(on_threshold=0.7, off_threshold=0.7)


def test_the_controller_reverses_far_less_than_the_instantaneous_one():
    rng = np.random.default_rng(1)
    util = np.clip(0.62 + 0.10 * np.sin(np.linspace(0, 6 * np.pi, 300))
                   + rng.normal(0, 0.035, 300), 0, 1)
    naive = [CTL.naive_surge(u) for u in util]
    ctl = CTL.SurgeController()
    smart = [ctl.observe(u) for u in util]
    assert CTL.oscillation(smart)["reversals"] < CTL.oscillation(naive)["reversals"]


def test_the_rate_limit_bounds_every_step():
    ctl = CTL.SurgeController(max_step=0.05)
    prev = ctl.multiplier
    for u in [0.0, 1.0] * 30:
        m = ctl.observe(u)
        assert abs(m - prev) <= 0.05 + 1e-9
        prev = m


def test_the_multiplier_never_exceeds_the_cap_or_drops_below_one():
    ctl = CTL.SurgeController(cap=2.0)
    for u in np.linspace(0, 1, 200):
        m = ctl.observe(float(u))
        assert 1.0 <= m <= 2.0


def test_sustained_slack_returns_the_multiplier_to_one():
    ctl = CTL.SurgeController()
    for _ in range(80):
        ctl.observe(0.95)
    assert ctl.multiplier > 1.0
    for _ in range(200):
        ctl.observe(0.2)
    assert ctl.multiplier == pytest.approx(1.0, abs=1e-6)


def test_the_cap_is_no_longer_inert():
    """The previous slope reached 1.88 at utilisation 1.0 against a 2.5 ceiling,
    so the cap shaped nothing and a test pinned that fact."""
    ctl = CTL.SurgeController(cap=1.5)
    for _ in range(200):
        ctl.observe(1.0)
    assert ctl.multiplier == pytest.approx(1.5, abs=1e-6)


def test_oscillation_counts_direction_changes():
    assert CTL.oscillation([1, 2, 1, 2, 1])["reversals"] == 3
    assert CTL.oscillation([1, 2, 3, 4])["reversals"] == 0


# --------------------------------------------------------------------------
# repositioning
# --------------------------------------------------------------------------
def test_idle_couriers_move_toward_value_and_busy_ones_do_not():
    rng = np.random.default_rng(0)
    zone = np.zeros(10, dtype=int)
    idle = np.array([True] * 5 + [False] * 5)
    demand = np.array([1.0, 20.0])
    supply = np.array([10.0, 1.0])
    travel = np.array([[0.0, 3.0], [3.0, 0.0]])
    out = CTL.reposition(zone, idle, demand, supply, travel, np.ones(2), rng,
                         compliance=1.0)
    assert (out[5:] == 0).all(), "busy couriers must not move"
    assert (out[:5] == 1).all(), "idle couriers should chase the value"


def test_compliance_below_one_leaves_some_couriers_put():
    rng = np.random.default_rng(2)
    zone = np.zeros(200, dtype=int)
    idle = np.ones(200, dtype=bool)
    demand = np.array([1.0, 20.0])
    supply = np.array([100.0, 1.0])
    travel = np.array([[0.0, 3.0], [3.0, 0.0]])
    out = CTL.reposition(zone, idle, demand, supply, travel, np.ones(2), rng,
                         compliance=0.5)
    moved = (out == 1).mean()
    assert 0.3 < moved < 0.7


def test_travel_time_damps_the_move():
    """The decision is a RATIO net of travel, so a distant zone has to be much
    better rather than merely better."""
    rng = np.random.default_rng(3)
    zone = np.zeros(50, dtype=int)
    idle = np.ones(50, dtype=bool)
    demand = np.array([10.0, 12.0])
    supply = np.array([10.0, 10.0])
    near = np.array([[0.0, 1.0], [1.0, 0.0]])
    far = np.array([[0.0, 90.0], [90.0, 0.0]])
    moved_near = (CTL.reposition(zone, idle, demand, supply, near, np.ones(2),
                                 rng, compliance=1.0) == 1).mean()
    moved_far = (CTL.reposition(zone, idle, demand, supply, far, np.ones(2),
                                rng, compliance=1.0) == 1).mean()
    assert moved_far <= moved_near


def test_herding_index_is_zero_when_even_and_one_when_concentrated():
    assert CTL.herding_index(np.array([5, 5, 5, 5])) == pytest.approx(0.0, abs=1e-9)
    assert CTL.herding_index(np.array([20, 0, 0, 0])) == pytest.approx(1.0, abs=1e-9)


def test_surge_raises_the_attractiveness_of_a_zone():
    rng = np.random.default_rng(4)
    zone = np.zeros(40, dtype=int)
    idle = np.ones(40, dtype=bool)
    demand = np.array([10.0, 10.0])
    supply = np.array([10.0, 10.0])
    travel = np.array([[0.0, 6.0], [6.0, 0.0]])
    flat = (CTL.reposition(zone, idle, demand, supply, travel, np.ones(2), rng,
                           compliance=1.0) == 1).mean()
    surged = (CTL.reposition(zone, idle, demand, supply, travel,
                             np.array([1.0, 2.5]), rng, compliance=1.0) == 1).mean()
    assert surged > flat
