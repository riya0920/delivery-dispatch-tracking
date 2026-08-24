"""Per-courier repositioning, and the ceiling that says whether it could have
helped at all.

The previous pass measured that perfect compliance buys nothing under a
broadcast, and named the broadcast as "the actual defect". Building the fix
showed that conclusion was scoped to a market too short for it to bite.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import control as CTL  # noqa: E402


DEMAND = np.array([3.0, 26.0, 2.0, 34.0, 4.0, 1.5, 18.0, 2.5])
TRAVEL = np.abs(np.arange(8)[:, None] - np.arange(8)[None, :]) * 3.0


def _simulate(fn, n_couriers, compliance=1.0, steps=200, seed=11):
    rng = np.random.default_rng(seed)
    zone = rng.integers(0, 8, n_couriers)
    ctrls = [CTL.SurgeController() for _ in range(8)]
    served = unserved = 0.0
    herd = []
    for _ in range(steps):
        counts = np.bincount(zone, minlength=8).astype(float)
        util = np.clip(DEMAND / np.maximum(counts, 1e-6) / 2.0, 0, 1)
        surge = np.array([c.observe(u) for c, u in zip(ctrls, util)])
        s = np.minimum(DEMAND, counts * 2.0)
        served += float(s.sum())
        unserved += float((DEMAND - s).sum())
        herd.append(CTL.herding_index(counts))
        if fn is not None:
            idle = rng.random(n_couriers) < 0.35
            zone = fn(zone, idle, DEMAND, counts, TRAVEL, surge, rng,
                      compliance=compliance)
    return served / (served + unserved), float(np.mean(herd))


# --------------------------------------------------------------------------
# the ceiling
# --------------------------------------------------------------------------
def test_the_oracle_is_one_only_when_every_ZONE_is_covered():
    """Couriers are indivisible and cannot be split across zones, so total
    capacity >= total demand is NOT enough: a zone wanting 2.5 orders of service
    needs two whole couriers and wastes one and a half.

    Total demand is 91, so 46 couriers carry 92 of capacity and still cannot
    reach 1.0. The sum of per-zone requirements is 47. This test asserted the
    first version and failed, which is the arithmetic being real rather than the
    oracle being wrong."""
    by_total = int(np.ceil(DEMAND.sum() / 2.0))
    by_zone = int(sum(np.ceil(DEMAND / 2.0)))
    assert by_zone > by_total
    assert CTL.oracle_fill(DEMAND, by_total, 2.0) < 1.0
    assert CTL.oracle_fill(DEMAND, by_zone, 2.0) == pytest.approx(1.0, abs=1e-9)


def test_the_oracle_is_monotone_in_fleet_size():
    prev = -1.0
    for n in range(0, 60, 3):
        f = CTL.oracle_fill(DEMAND, n, 2.0)
        assert f >= prev - 1e-12
        prev = f


def test_the_oracle_never_promises_more_than_the_fleet_can_serve():
    for n in (5, 12, 29, 40):
        assert CTL.oracle_fill(DEMAND, n, 2.0) <= n * 2.0 / DEMAND.sum() + 1e-9


def test_no_policy_can_beat_the_oracle():
    """The whole point of having it. A policy above the ceiling means the
    ceiling is wrong, and the ceiling is what every conclusion in the section
    rests on."""
    for n in (29, 40, 54):
        ceiling = CTL.oracle_fill(DEMAND, n, 2.0)
        for fn in (None, CTL.reposition, CTL.reposition_targeted):
            fill, _ = _simulate(fn, n)
            assert fill <= ceiling + 1e-9, (n, fn, fill, ceiling)


# --------------------------------------------------------------------------
# the targeted policy
# --------------------------------------------------------------------------
def test_a_zone_is_offered_to_as_many_couriers_as_it_is_short():
    """The capacity cap is the whole mechanism: an argmax everybody can compute
    for themselves is not a recommendation, it is an announcement."""
    rng = np.random.default_rng(0)
    zone = np.zeros(40, dtype=int)
    idle = np.ones(40, dtype=bool)
    demand = np.array([2.0, 12.0])          # zone 1 wants 6 couriers
    supply = np.array([40.0, 0.0])
    travel = np.array([[0.0, 3.0], [3.0, 0.0]])
    out = CTL.reposition_targeted(zone, idle, demand, supply, travel,
                                  np.ones(2), rng, served_per_courier=2.0,
                                  compliance=1.0)
    moved = int((out == 1).sum())
    assert moved == 6, moved


def test_the_broadcast_sends_everybody_to_the_same_zone():
    """The contrast that makes the cap meaningful."""
    rng = np.random.default_rng(0)
    zone = np.zeros(40, dtype=int)
    idle = np.ones(40, dtype=bool)
    demand = np.array([2.0, 12.0])
    supply = np.array([40.0, 0.0])
    travel = np.array([[0.0, 3.0], [3.0, 0.0]])
    out = CTL.reposition(zone, idle, demand, supply, travel, np.ones(2), rng,
                         compliance=1.0)
    assert int((out == 1).sum()) == 40


def test_a_declined_offer_does_not_consume_the_slot():
    """Otherwise low compliance looks worse for a modelling reason rather than a
    behavioural one, and the conclusion would have been built into the setup."""
    demand = np.array([2.0, 12.0])
    supply = np.array([200.0, 0.0])
    travel = np.array([[0.0, 3.0], [3.0, 0.0]])
    zone = np.zeros(200, dtype=int)
    idle = np.ones(200, dtype=bool)
    moved = []
    for seed in range(12):
        rng = np.random.default_rng(seed)
        out = CTL.reposition_targeted(zone, idle, demand, supply, travel,
                                      np.ones(2), rng, served_per_courier=2.0,
                                      compliance=0.5)
        moved.append(int((out == 1).sum()))
    # the deficit is 6; with re-offers it should fill nearly every time
    assert np.mean(moved) > 5.0, moved


def test_busy_couriers_are_never_moved():
    rng = np.random.default_rng(1)
    zone = np.zeros(20, dtype=int)
    idle = np.array([True] * 10 + [False] * 10)
    demand = np.array([1.0, 30.0])
    supply = np.array([20.0, 0.0])
    travel = np.array([[0.0, 3.0], [3.0, 0.0]])
    out = CTL.reposition_targeted(zone, idle, demand, supply, travel,
                                  np.ones(2), rng, compliance=1.0)
    assert (out[10:] == 0).all()


def test_a_zone_with_no_deficit_attracts_nobody():
    rng = np.random.default_rng(2)
    zone = np.zeros(30, dtype=int)
    idle = np.ones(30, dtype=bool)
    demand = np.array([1.0, 4.0])
    supply = np.array([30.0, 10.0])        # zone 1 already over-supplied
    travel = np.array([[0.0, 3.0], [3.0, 0.0]])
    out = CTL.reposition_targeted(zone, idle, demand, supply, travel,
                                  np.ones(2), rng, compliance=1.0)
    assert (out == 0).all()


# --------------------------------------------------------------------------
# the regime finding
# --------------------------------------------------------------------------
def test_in_a_scarce_market_the_broadcast_is_already_at_the_ceiling():
    """Which is why the previous pass could not see the defect it named: with
    demand this concentrated and capacity at 64% of it, concentration IS the
    optimum and the herding index is measuring correct behaviour."""
    n = int(DEMAND.sum() * 0.65 / 2.0)
    ceiling = CTL.oracle_fill(DEMAND, n, 2.0)
    broadcast, _ = _simulate(CTL.reposition, n)
    assert ceiling - broadcast < 0.01, (ceiling, broadcast)


def test_once_supply_covers_demand_targeting_recovers_what_the_broadcast_gives_up():
    n = int(DEMAND.sum() * 1.20 / 2.0)
    ceiling = CTL.oracle_fill(DEMAND, n, 2.0)
    broadcast, b_herd = _simulate(CTL.reposition, n)
    targeted, t_herd = _simulate(CTL.reposition_targeted, n)
    headroom = ceiling - broadcast
    assert headroom > 0.02, headroom
    assert (targeted - broadcast) / headroom > 0.8
    assert t_herd < b_herd


def test_targeting_lowers_herding_without_lowering_fill():
    """The capacity cap stops the herd by refusing to make the same
    recommendation twice, not by asking couriers to be less obedient."""
    for ratio in (0.65, 1.20):
        n = int(DEMAND.sum() * ratio / 2.0)
        broadcast, b_herd = _simulate(CTL.reposition, n)
        targeted, t_herd = _simulate(CTL.reposition_targeted, n)
        assert t_herd <= b_herd + 1e-9, (ratio, t_herd, b_herd)
        assert targeted >= broadcast - 0.005, (ratio, targeted, broadcast)
