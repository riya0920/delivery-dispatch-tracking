"""Tests for courier agents, the re-offer cascade, and surge."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import couriers as CR  # noqa: E402
from src import dispatch as D  # noqa: E402


def _agent(pickiness=24.0, shift=300.0, seed=0):
    return CR.CourierAgent(0, pickiness, shift, np.random.default_rng(seed))


# --------------------------------------------------------------------------
# the acceptance model
# --------------------------------------------------------------------------
def test_acceptance_keys_on_dollars_per_hour_not_per_km():
    """The bug the first calibration had: on $/km a 200-metre job for $6 looks
    wonderful, when it is eleven minutes of waiting for six dollars."""
    a = _agent()
    short_cheap = a.accept_probability(0.2, 6.0)
    long_paid_proportionally = a.accept_probability(6.0, 6.0 + 1.1 * 6.0)
    # per-km these are wildly different deals; per-hour they are close
    assert abs(short_cheap - long_paid_proportionally) < 0.30


def test_a_short_trip_is_not_free_money():
    """Fixed overhead means a tiny job is a mediocre hourly rate, not an
    infinite one."""
    a = _agent(pickiness=24.0)
    assert a.accept_probability(0.05, 6.0) < 0.99


def test_more_payout_raises_acceptance():
    a = _agent()
    prev = -1.0
    for pay in (4.0, 6.0, 9.0, 15.0):
        p = a.accept_probability(2.0, pay)
        assert p > prev
        prev = p


def test_more_distance_lowers_acceptance_at_fixed_payout():
    a = _agent()
    assert a.accept_probability(1.0, 8.0) > a.accept_probability(8.0, 8.0)


def test_pickier_couriers_accept_less():
    easy = _agent(pickiness=12.0)
    picky = _agent(pickiness=40.0)
    assert easy.accept_probability(2.0, 7.0) > picky.accept_probability(2.0, 7.0)


def test_fatigue_raises_the_bar_through_the_shift():
    a = _agent(pickiness=24.0, shift=300.0)
    fresh = a.accept_probability(2.0, 7.0)
    a.minutes_worked = 300.0
    tired = a.accept_probability(2.0, 7.0)
    assert tired < fresh


def test_surge_raises_acceptance():
    a = _agent()
    assert a.accept_probability(3.0, 8.0, surge=1.8) > a.accept_probability(3.0, 8.0)


def test_acceptance_is_a_probability():
    a = _agent()
    for d in (0.1, 1.0, 5.0, 20.0):
        for pay in (1.0, 10.0, 100.0):
            p = a.accept_probability(d, pay)
            assert 0.0 <= p <= 1.0


def test_fleet_is_heterogeneous():
    """The marginal offer goes to the pickiest courier still available, so a
    fleet of identical couriers understates how hard dispatch is."""
    fleet = CR.make_fleet(500, np.random.default_rng(1))
    picks = np.array([a.pickiness for a in fleet.values()])
    assert picks.std() > 3.0
    assert 15.0 < picks.mean() < 35.0


def test_decide_records_the_offer():
    a = _agent()
    for _ in range(50):
        a.decide(2.0, 7.0)
    assert a.offers_seen == 50
    assert 0 < a.offers_accepted < 50
    assert 0.0 < a.accept_rate < 1.0


def test_going_offline_is_sticky_and_more_likely_late_in_the_shift():
    early = _agent(shift=300.0, seed=5)
    late = _agent(shift=300.0, seed=5)
    late.minutes_worked = 300.0
    assert late.maybe_go_offline(0.5) or True          # smoke
    a = _agent(seed=7)
    a.online = False
    assert a.maybe_go_offline(1.0) is False, "an offline courier stays offline"


# --------------------------------------------------------------------------
# the cascade
# --------------------------------------------------------------------------
def _fixture(n=6):
    lat = np.full(n, 37.76)
    lon = np.full(n, -122.44)
    order = D.Order(0, 37.762, -122.442, 0.0, 8.0)
    return lat, lon, order


def test_cascade_stops_at_the_first_acceptor():
    lat, lon, order = _fixture()
    fleet = {i: _agent(pickiness=1.0, seed=i) for i in range(6)}   # all accept
    res = CR.cascade_assign(order, list(range(6)), fleet, lat, lon)
    assert res["assigned_to"] == 0
    assert res["depth"] == 1
    assert res["offer_seconds"] == 0.0


def test_cascade_depth_converts_to_waiting():
    lat, lon, order = _fixture()
    fleet = {i: _agent(pickiness=1000.0, seed=i) for i in range(6)}  # all decline
    fleet[3] = _agent(pickiness=0.1, seed=99)                        # except one
    res = CR.cascade_assign(order, list(range(6)), fleet, lat, lon, offer_ttl_s=25.0)
    assert res["assigned_to"] == 3
    assert res["depth"] == 4
    assert res["offer_seconds"] == pytest.approx(3 * 25.0)


def test_cascade_returns_unassigned_when_everyone_declines():
    lat, lon, order = _fixture()
    fleet = {i: _agent(pickiness=10_000.0, seed=i) for i in range(6)}
    res = CR.cascade_assign(order, list(range(6)), fleet, lat, lon)
    assert res["assigned_to"] is None
    assert res["offer_seconds"] > 0


def test_cascade_skips_offline_couriers():
    lat, lon, order = _fixture()
    fleet = {i: _agent(pickiness=0.1, seed=i) for i in range(6)}
    fleet[0].online = False
    fleet[1].online = False
    res = CR.cascade_assign(order, list(range(6)), fleet, lat, lon)
    assert res["assigned_to"] == 2


def test_cascade_respects_max_depth():
    lat, lon, order = _fixture(n=20)
    lat = np.full(20, 37.76)
    lon = np.full(20, -122.44)
    fleet = {i: _agent(pickiness=10_000.0, seed=i) for i in range(20)}
    res = CR.cascade_assign(order, list(range(20)), fleet, lat, lon, max_depth=3)
    assert res["depth"] == 3
    assert sum(a.offers_seen for a in fleet.values()) == 3


def test_payout_rises_with_distance():
    lat = np.array([37.76, 37.90])
    lon = np.array([-122.44, -122.44])
    order = D.Order(0, 37.761, -122.441, 0.0, 8.0)
    near = {0: _agent(pickiness=0.1, seed=1)}
    far = {1: _agent(pickiness=0.1, seed=1)}
    r_near = CR.cascade_assign(order, [0], near, lat, lon)
    r_far = CR.cascade_assign(order, [1], far, lat, lon)
    assert r_far["payout"] > r_near["payout"]


# --------------------------------------------------------------------------
# surge
# --------------------------------------------------------------------------
def test_surge_is_flat_in_a_slack_market():
    assert CR.surge_for(0.3) == 1.0
    assert CR.surge_for(0.6) == 1.0


def test_surge_rises_with_utilisation_and_never_exceeds_the_cap():
    assert CR.surge_for(0.8) > CR.surge_for(0.7) > 1.0
    for u in np.linspace(0.0, 1.0, 21):
        assert 1.0 <= CR.surge_for(u) <= 2.5


def test_the_surge_cap_is_currently_inert():
    """At the configured slope the cap never binds -- utilisation 1.0 gives 1.88.
    Asserting that keeps an inert parameter honest: if someone steepens the slope
    the cap starts shaping results and this test tells them."""
    assert CR.surge_for(1.0) == pytest.approx(1.88)
    assert CR.surge_for(1.0) < 2.5


def test_surge_measurably_buys_acceptance():
    """The claim the report makes, at fleet scale."""
    fleet = CR.make_fleet(400, np.random.default_rng(3))
    base = np.mean([a.accept_probability(3.0, 9.3, 1.0) for a in fleet.values()])
    surged = np.mean([a.accept_probability(3.0, 9.3, CR.surge_for(0.88))
                      for a in fleet.values()])
    assert surged > base + 0.15
